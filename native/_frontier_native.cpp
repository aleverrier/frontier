#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxLimbs = 64;
constexpr int kMaxLogicalLimbs = 8;
constexpr int kCompactLimbs = 4;
constexpr int kSmallPatternRows = 6;
constexpr std::size_t kSmallPatternTableSize = 1ULL << kSmallPatternRows;
constexpr std::size_t kLinearMergeLimit = 4;
constexpr std::size_t kActive4SmallStateStepLimit = 16;
constexpr std::size_t kOnePassPruneMinCandidates = 4096;
std::size_t merge_bucket_count_for(std::size_t expected_candidates) {
    std::size_t count = 32;
    const std::size_t target = expected_candidates * 4 + 16;
    while (count < target) {
        count <<= 1U;
    }
    return count;
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

double logaddexp_pair(double a, double b) {
    if (!std::isfinite(a)) return b;
    if (!std::isfinite(b)) return a;
    const double hi = std::max(a, b);
    const double lo = std::min(a, b);
    return hi + std::log1p(std::exp(lo - hi));
}

enum class MetricMode {
    LogSumExpFloat,
    MaxLogInt,
};

constexpr std::int64_t kIntMetricNegInf = std::numeric_limits<std::int64_t>::min() / 4;

MetricMode parse_metric_mode(const char* raw) {
    const std::string value(raw == nullptr ? "logsumexp_float" : raw);
    if (value == "logsumexp_float" || value == "float" || value == "exact") {
        return MetricMode::LogSumExpFloat;
    }
    if (
        value == "frontierLite" || value == "frontier_lite" || value == "frontier-lite"
        || value == "frontierlite" || value == "maxlog_int" || value == "max_log_int"
        || value == "viterbi_int"
    ) {
        return MetricMode::MaxLogInt;
    }
    throw std::runtime_error("metric_mode must be 'logsumexp_float', 'frontier_lite', or 'maxlog_int'");
}

void validate_metric_options(MetricMode mode, int int_metric_scale) {
    if (mode == MetricMode::MaxLogInt && int_metric_scale <= 0) {
        throw std::runtime_error("int_metric_scale must be positive for frontierLite/maxlog_int");
    }
}

std::int64_t quantize_metric(double value, int int_metric_scale) {
    if (!std::isfinite(value)) return kIntMetricNegInf;
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(int_metric_scale);
    const long double lo = static_cast<long double>(kIntMetricNegInf + 1);
    const long double hi = static_cast<long double>(std::numeric_limits<std::int64_t>::max() / 4);
    if (scaled <= lo) return kIntMetricNegInf;
    if (scaled >= hi) return std::numeric_limits<std::int64_t>::max() / 4;
    return static_cast<std::int64_t>(std::llround(static_cast<double>(scaled)));
}

std::int64_t fixed_mul_round(std::int64_t value, std::int64_t multiplier, std::int64_t scale) {
    if (value <= kIntMetricNegInf / 2) return kIntMetricNegInf;
    if (multiplier == 0) return 0;
    if (multiplier == scale) return value;
    __int128 product = static_cast<__int128>(value) * static_cast<__int128>(multiplier);
    const __int128 divisor = static_cast<__int128>(scale);
    if (product >= 0) {
        product += divisor / 2;
        return static_cast<std::int64_t>(product / divisor);
    }
    product = -product + divisor / 2;
    return -static_cast<std::int64_t>(product / divisor);
}

std::int64_t fixed_mul_round_fast(std::int64_t value, std::int64_t multiplier, std::int64_t scale) {
    if (value <= kIntMetricNegInf / 2) return kIntMetricNegInf;
    if (multiplier == 0) return 0;
    if (multiplier == scale) return value;
    if (scale == 1024) {
        __int128 product = static_cast<__int128>(value) * static_cast<__int128>(multiplier);
        if (product >= 0) {
            product += 512;
            return static_cast<std::int64_t>(product >> 10);
        }
        product = -product + 512;
        return -static_cast<std::int64_t>(product >> 10);
    }
    return fixed_mul_round(value, multiplier, scale);
}

std::int64_t score_int_metric(
    std::int64_t logmass,
    std::int64_t parity_logodds,
    std::int64_t alpha_int,
    int int_metric_scale
) {
    const std::int64_t scaled_parity = fixed_mul_round_fast(parity_logodds, alpha_int, int_metric_scale);
    if (scaled_parity <= kIntMetricNegInf / 2) return kIntMetricNegInf;
    return logmass + scaled_parity;
}

std::int64_t quantize_metric_s1024_cached(double value, std::int64_t cached_s1024, int int_metric_scale) {
    if (int_metric_scale == 1024) return cached_s1024;
    return quantize_metric(value, int_metric_scale);
}

bool batch_workspace_reuse_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_BATCH_WORKSPACE_REUSE");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::size_t native_batch_thread_count(std::size_t job_count) {
    if (job_count <= 1) return 1;
    const char* raw = std::getenv("FRONTIER_NATIVE_BATCH_THREADS");
    if (raw == nullptr) return 1;
    const std::string value(raw);
    if (value.empty() || value == "1" || value == "off" || value == "false" || value == "no") {
        return 1;
    }
    std::size_t requested = 1;
    if (value == "auto" || value == "on" || value == "true" || value == "yes") {
        requested = static_cast<std::size_t>(std::max(1U, std::thread::hardware_concurrency()));
    } else {
        try {
            requested = static_cast<std::size_t>(std::stoul(value));
        } catch (...) {
            return 1;
        }
    }
    if (requested <= 1) return 1;
    return std::min(requested, job_count);
}

bool one_pass_prune_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_ONE_PASS_PRUNE");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::size_t one_pass_prune_min_candidates() {
    const char* raw = std::getenv("FRONTIER_NATIVE_ONE_PASS_PRUNE_MIN");
    if (raw == nullptr) return kOnePassPruneMinCandidates;
    try {
        return std::max<std::size_t>(1U, static_cast<std::size_t>(std::stoul(std::string(raw))));
    } catch (...) {
        return kOnePassPruneMinCandidates;
    }
}

bool final_prune_sort_disabled() {
    const char* force_sort_raw = std::getenv("FRONTIER_NATIVE_ENABLE_FINAL_PRUNE_SORT");
    if (force_sort_raw != nullptr) {
        const std::string force_value(force_sort_raw);
        if (force_value == "1" || force_value == "true" || force_value == "on" || force_value == "yes") {
            return false;
        }
    }
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_FINAL_PRUNE_SORT");
    if (raw == nullptr) return true;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool close_empty_split_merge_disabled() {
    const char* enable_raw = std::getenv("FRONTIER_NATIVE_ENABLE_CLOSE_EMPTY_SPLIT_MERGE");
    if (enable_raw == nullptr) return true;
    const std::string enable_value(enable_raw);
    if (!(enable_value == "1" || enable_value == "true" || enable_value == "on" || enable_value == "yes")) {
        return true;
    }
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_CLOSE_EMPTY_SPLIT_MERGE");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool compact_close_empty_split_merge_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_COMPACT_CLOSE_EMPTY_SPLIT_MERGE");
    if (raw != nullptr) {
        const std::string value(raw);
        return value == "1" || value == "true" || value == "on" || value == "yes";
    }
    return false;
}

bool no_merge_transition_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_NO_MERGE_TRANSITION");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool single_parent_step_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_SINGLE_PARENT_STEP");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool small_state_step_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_SMALL_STATE_STEP");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool native_profile_enabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_PROFILE");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool small_pattern_table_disabled() {
    const char* raw = std::getenv("FRONTIER_NATIVE_DISABLE_SMALL_PATTERN_TABLE");
    if (raw == nullptr) return false;
    const std::string value(raw);
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool dict_set_steal(PyObject* dict, const char* name, PyObject* value) {
    if (value == nullptr) return false;
    const int status = PyDict_SetItemString(dict, name, value);
    Py_DECREF(value);
    return status == 0;
}

using LogicalMask = std::array<std::uint64_t, kMaxLogicalLimbs>;

std::uint64_t mix_hash_word(std::uint64_t value);

LogicalMask logical_xor(const LogicalMask& lhs, const LogicalMask& rhs, int n_logical_limbs) {
    LogicalMask out{};
    for (int index = 0; index < n_logical_limbs; ++index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        out[limb_index] = lhs[limb_index] ^ rhs[limb_index];
    }
    return out;
}

void logical_xor_inplace(LogicalMask& lhs, const LogicalMask& rhs, int n_logical_limbs) {
    for (int index = 0; index < n_logical_limbs; ++index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        lhs[limb_index] ^= rhs[limb_index];
    }
}

bool logical_equal_limited(const LogicalMask& lhs, const LogicalMask& rhs, int n_logical_limbs) {
    for (int index = 0; index < n_logical_limbs; ++index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        if (lhs[limb_index] != rhs[limb_index]) return false;
    }
    return true;
}

bool logical_less_limited(const LogicalMask& lhs, const LogicalMask& rhs, int n_logical_limbs) {
    for (int index = 0; index < n_logical_limbs; ++index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        if (lhs[limb_index] != rhs[limb_index]) {
            return lhs[limb_index] < rhs[limb_index];
        }
    }
    return false;
}

std::size_t hash_logical(const LogicalMask& logical, int n_logical_limbs) {
    std::uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int index = 0; index < n_logical_limbs; ++index) {
        const std::uint64_t value = mix_hash_word(logical[static_cast<std::size_t>(index)]);
        h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<std::size_t>(h);
}

struct Key {
    std::array<std::uint64_t, kMaxLimbs> det;
    LogicalMask logical{};
};

struct KeyLess {
    int n_limbs = 0;

    bool operator()(const Key& lhs, const Key& rhs) const {
        if (lhs.logical != rhs.logical) return lhs.logical < rhs.logical;
        for (int index = n_limbs - 1; index >= 0; --index) {
            if (lhs.det[static_cast<std::size_t>(index)] != rhs.det[static_cast<std::size_t>(index)]) {
                return lhs.det[static_cast<std::size_t>(index)] < rhs.det[static_cast<std::size_t>(index)];
            }
        }
        return false;
    }
};

struct KeyEqual {
    int n_limbs = 0;

    bool operator()(const Key& lhs, const Key& rhs) const {
        if (lhs.logical != rhs.logical) return false;
        for (int index = 0; index < n_limbs; ++index) {
            if (lhs.det[static_cast<std::size_t>(index)] != rhs.det[static_cast<std::size_t>(index)]) {
                return false;
            }
        }
        return true;
    }
};

bool det_zero(const Key& key, int n_limbs) {
    for (int index = 0; index < n_limbs; ++index) {
        if (key.det[static_cast<std::size_t>(index)] != 0) return false;
    }
    return true;
}

struct RowTerm {
    int limb = 0;
    int active4_slot = -1;
    std::uint64_t bit = 0;
    double parity = 0.0;
    std::int64_t parity_s1024 = 0;
};

struct LocalPatternTable {
    bool enabled = false;
    bool small_enabled = false;
    int row_count = 0;
    std::array<RowTerm, 12> rows{};
    std::array<double, kSmallPatternTableSize> no_delta_small{};
    std::array<double, kSmallPatternTableSize> toggle_delta_small{};
    std::array<std::int64_t, kSmallPatternTableSize> no_delta_small_s1024{};
    std::array<std::int64_t, kSmallPatternTableSize> toggle_delta_small_s1024{};
    std::vector<double> no_delta;
    std::vector<double> toggle_delta;
    std::vector<std::int64_t> no_delta_s1024;
    std::vector<std::int64_t> toggle_delta_s1024;
};

struct Column {
    double no_error_log_const = 0.0;
    double toggle_logodds = -std::numeric_limits<double>::infinity();
    std::int64_t no_error_log_const_s1024 = 0;
    std::int64_t toggle_logodds_s1024 = kIntMetricNegInf;
    std::array<std::uint64_t, kMaxLimbs> toggle_detector{};
    std::array<std::uint64_t, kMaxLimbs> active_toggle_detector{};
    std::array<std::uint64_t, kMaxLimbs> close_mask{};
    std::array<std::uint64_t, kMaxLimbs> active_mask{};
    LogicalMask toggle_logical{};
    std::vector<RowTerm> before_terms;
    std::vector<RowTerm> after_terms;
    std::array<int, kCompactLimbs> before_active_limbs4{};
    std::array<int, kMaxLimbs> before_slot_by_limb4{};
    std::array<int, kCompactLimbs> active_before_slots4{};
    std::array<std::uint64_t, kCompactLimbs> active_slot_masks4{};
    std::array<std::uint64_t, kCompactLimbs> active_toggle_slots4{};
    std::array<int, kMaxLimbs> close_nonzero_limbs{};
    std::array<int, kMaxLimbs> active_nonzero_limbs{};
    std::array<int, kMaxLimbs> toggle_nonzero_limbs{};
    int before_active_count4 = 0;
    int close_nonzero_count = 0;
    int active_nonzero_count = 0;
    int toggle_nonzero_count = 0;
    bool no_child_injective4 = false;
    bool active4_no_child_identity = false;
    bool toggle_has_new_active_bit = false;
    LocalPatternTable pattern_table;
    bool sparse_clear_supported = false;
};

std::uint64_t mix_hash_word(std::uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

std::size_t hash_key_for_column(const Key& key, const Column& column, int n_limbs, int n_logical_limbs) {
    std::uint64_t h = static_cast<std::uint64_t>(hash_logical(key.logical, n_logical_limbs));
    if (column.sparse_clear_supported) {
        for (int offset = 0; offset < column.active_nonzero_count; ++offset) {
            const int index = column.active_nonzero_limbs[static_cast<std::size_t>(offset)];
            const std::uint64_t value = mix_hash_word(key.det[static_cast<std::size_t>(index)]);
            h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<std::size_t>(h);
    }
    for (int index = 0; index < n_limbs; ++index) {
        const std::uint64_t value = mix_hash_word(key.det[static_cast<std::size_t>(index)]);
        h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<std::size_t>(h);
}

bool key_equal_for_column(const Key& lhs, const Key& rhs, const Column& column, int n_limbs) {
    if (lhs.logical != rhs.logical) return false;
    if (column.sparse_clear_supported) {
        for (int offset = 0; offset < column.active_nonzero_count; ++offset) {
            const int index = column.active_nonzero_limbs[static_cast<std::size_t>(offset)];
            if (lhs.det[static_cast<std::size_t>(index)] != rhs.det[static_cast<std::size_t>(index)]) {
                return false;
            }
        }
        return true;
    }
    for (int index = 0; index < n_limbs; ++index) {
        if (lhs.det[static_cast<std::size_t>(index)] != rhs.det[static_cast<std::size_t>(index)]) {
            return false;
        }
    }
    return true;
}

struct State {
    Key key;
    double logmass = -std::numeric_limits<double>::infinity();
    double parity_logodds = 0.0;
};

struct Candidate {
    Key key;
    double logmass = -std::numeric_limits<double>::infinity();
    double parity_logodds = 0.0;
    double score = -std::numeric_limits<double>::infinity();
};

struct NativeModel {
    int num_detectors = 0;
    int num_observables = 0;
    int n_limbs = 0;
    int n_logical_limbs = 0;
    bool collect_phase_timing = false;
    bool force_full_key = false;
    bool active4_supported = true;
    std::vector<Column> columns;
};

struct ChoiceOption {
    double log_prior = -std::numeric_limits<double>::infinity();
    std::array<std::uint64_t, kMaxLimbs> detector{};
    LogicalMask logical{};
};

struct ChoiceColumn {
    std::array<std::uint64_t, kMaxLimbs> close_mask{};
    std::array<std::uint64_t, kMaxLimbs> active_mask{};
    std::array<int, kMaxLimbs> close_nonzero_limbs{};
    int close_nonzero_count = 0;
    std::vector<RowTerm> before_terms;
    std::vector<RowTerm> after_terms;
    std::vector<ChoiceOption> options;
};

struct ChoiceNativeModel {
    int num_detectors = 0;
    int num_observables = 0;
    int n_limbs = 0;
    int n_logical_limbs = 0;
    bool collect_phase_timing = false;
    std::vector<ChoiceColumn> columns;
};

struct DecodeStats {
    int processed_columns = 0;
    std::uint64_t transition_evals = 0;
    std::uint64_t max_pre_prune_state_count = 0;
    std::uint64_t max_post_prune_state_count = 0;
    std::uint64_t sum_pre_prune_state_count = 0;
    std::uint64_t sum_post_prune_state_count = 0;
    int no_path_count = 0;
    double transition_time_s = 0.0;
    double merge_time_s = 0.0;
    double prune_time_s = 0.0;
    double total_time_s = 0.0;
    std::uint64_t profile_no_merge_transition_columns = 0;
    std::uint64_t profile_split_merge_columns = 0;
    std::uint64_t profile_generic_merge_columns = 0;
    std::uint64_t profile_emit_child_calls = 0;
    std::uint64_t profile_merge_duplicate_count = 0;
    std::uint64_t profile_hash_probe_total = 0;
    std::uint64_t profile_hash_probe_max = 0;
    std::uint64_t profile_score_evals = 0;
    std::uint64_t profile_nth_element_calls = 0;
};

struct DecodeResult {
    bool ok = false;
    LogicalMask logical_hat{};
    double log_evidence = -std::numeric_limits<double>::infinity();
    std::map<LogicalMask, double> terminal_log_masses;
    double terminal_top_log_mass_gap = std::numeric_limits<double>::quiet_NaN();
    DecodeStats stats;
};

PyObject* get_item(PyObject* dict, const char* name) {
    PyObject* value = PyDict_GetItemString(dict, name);
    if (value == nullptr) {
        throw std::runtime_error(std::string("missing native model field: ") + name);
    }
    return value;
}

long as_long(PyObject* obj, const char* name) {
    long value = PyLong_AsLong(obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error(std::string("invalid integer field: ") + name);
    }
    return value;
}

unsigned long long as_ull(PyObject* obj, const char* name) {
    unsigned long long value = PyLong_AsUnsignedLongLong(obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error(std::string("invalid uint64 field: ") + name);
    }
    return value;
}

double as_double(PyObject* obj, const char* name) {
    double value = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error(std::string("invalid float field: ") + name);
    }
    return value;
}

std::vector<std::uint64_t> parse_u64_list(PyObject* obj, const char* name) {
    PyObject* seq = PySequence_Fast(obj, name);
    if (seq == nullptr) {
        throw std::runtime_error(std::string("invalid sequence field: ") + name);
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    std::vector<std::uint64_t> out;
    out.reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t index = 0; index < size; ++index) {
        out.push_back(static_cast<std::uint64_t>(as_ull(items[index], name)));
    }
    Py_DECREF(seq);
    return out;
}

std::vector<double> parse_double_list(PyObject* obj, const char* name) {
    PyObject* seq = PySequence_Fast(obj, name);
    if (seq == nullptr) {
        throw std::runtime_error(std::string("invalid sequence field: ") + name);
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t index = 0; index < size; ++index) {
        out.push_back(as_double(items[index], name));
    }
    Py_DECREF(seq);
    return out;
}

std::array<std::uint64_t, kMaxLimbs> parse_limbs(PyObject* obj, int n_limbs, const char* name) {
    std::vector<std::uint64_t> values = parse_u64_list(obj, name);
    if (static_cast<int>(values.size()) != n_limbs) {
        throw std::runtime_error(std::string("wrong limb count for field: ") + name);
    }
    std::array<std::uint64_t, kMaxLimbs> out{};
    for (int index = 0; index < n_limbs; ++index) {
        out[static_cast<std::size_t>(index)] = values[static_cast<std::size_t>(index)];
    }
    return out;
}

LogicalMask parse_logical_limbs(PyObject* obj, int n_logical_limbs, const char* name) {
    std::vector<std::uint64_t> values = parse_u64_list(obj, name);
    if (static_cast<int>(values.size()) != n_logical_limbs) {
        throw std::runtime_error(std::string("wrong logical limb count for field: ") + name);
    }
    LogicalMask out{};
    for (int index = 0; index < n_logical_limbs; ++index) {
        out[static_cast<std::size_t>(index)] = values[static_cast<std::size_t>(index)];
    }
    return out;
}

PyObject* logical_to_pylong(const LogicalMask& logical, int n_logical_limbs) {
    PyObject* value = PyLong_FromLong(0);
    if (value == nullptr) return nullptr;
    for (int index = n_logical_limbs - 1; index >= 0; --index) {
        PyObject* shift_amount = PyLong_FromLong(64);
        if (shift_amount == nullptr) {
            Py_DECREF(value);
            return nullptr;
        }
        PyObject* shifted = PyNumber_Lshift(value, shift_amount);
        Py_DECREF(shift_amount);
        Py_DECREF(value);
        if (shifted == nullptr) return nullptr;
        PyObject* limb = PyLong_FromUnsignedLongLong(logical[static_cast<std::size_t>(index)]);
        if (limb == nullptr) {
            Py_DECREF(shifted);
            return nullptr;
        }
        value = PyNumber_Or(shifted, limb);
        Py_DECREF(shifted);
        Py_DECREF(limb);
        if (value == nullptr) return nullptr;
    }
    return value;
}

std::vector<RowTerm> parse_row_terms(PyObject* column_dict, const char* prefix) {
    const std::string limb_name = std::string(prefix) + "_limbs";
    const std::string bit_name = std::string(prefix) + "_bits";
    const std::string parity_name = std::string(prefix) + "_parity";
    std::vector<std::uint64_t> limbs = parse_u64_list(get_item(column_dict, limb_name.c_str()), limb_name.c_str());
    std::vector<std::uint64_t> bits = parse_u64_list(get_item(column_dict, bit_name.c_str()), bit_name.c_str());
    std::vector<double> parity = parse_double_list(get_item(column_dict, parity_name.c_str()), parity_name.c_str());
    if (limbs.size() != bits.size() || limbs.size() != parity.size()) {
        throw std::runtime_error("row-term arrays have mismatched lengths");
    }
    std::vector<RowTerm> out;
    out.reserve(limbs.size());
    for (std::size_t index = 0; index < limbs.size(); ++index) {
        RowTerm term;
        term.limb = static_cast<int>(limbs[index]);
        term.bit = bits[index];
        term.parity = parity[index];
        term.parity_s1024 = quantize_metric(term.parity, 1024);
        out.push_back(term);
    }
    return out;
}

int fill_nonzero_limbs(
    const std::array<std::uint64_t, kMaxLimbs>& values,
    int n_limbs,
    std::array<int, kMaxLimbs>& out
) {
    int count = 0;
    for (int index = 0; index < n_limbs; ++index) {
        if (values[static_cast<std::size_t>(index)] != 0) {
            out[static_cast<std::size_t>(count)] = index;
            count += 1;
        }
    }
    return count;
}

LocalPatternTable compile_local_pattern_table(
    const std::vector<RowTerm>& before_terms,
    const std::vector<RowTerm>& after_terms
) {
    LocalPatternTable table;
    std::map<std::pair<int, std::uint64_t>, std::pair<double, double>> weights;
    for (const RowTerm& term : before_terms) {
        weights[{term.limb, term.bit}].first += term.parity;
    }
    for (const RowTerm& term : after_terms) {
        weights[{term.limb, term.bit}].second += term.parity;
    }
    if (weights.empty() || weights.size() > 12) {
        return table;
    }
    table.enabled = true;
    table.row_count = static_cast<int>(weights.size());
    std::vector<double> before_weights;
    std::vector<double> after_weights;
    before_weights.reserve(weights.size());
    after_weights.reserve(weights.size());
    std::size_t row_index = 0;
    for (const auto& item : weights) {
        RowTerm term;
        term.limb = item.first.first;
        term.active4_slot = -1;
        term.bit = item.first.second;
        term.parity = 0.0;
        term.parity_s1024 = 0;
        table.rows[row_index] = term;
        before_weights.push_back(item.second.first);
        after_weights.push_back(item.second.second);
        row_index += 1U;
    }

    const std::size_t size = static_cast<std::size_t>(1ULL << static_cast<std::size_t>(table.row_count));
    table.small_enabled = table.row_count <= kSmallPatternRows && !small_pattern_table_disabled();
    if (!table.small_enabled) {
        table.no_delta.assign(size, 0.0);
        table.toggle_delta.assign(size, 0.0);
        table.no_delta_s1024.assign(size, 0);
        table.toggle_delta_s1024.assign(size, 0);
    }
    for (std::size_t pattern = 0; pattern < size; ++pattern) {
        double no_delta = 0.0;
        double toggle_delta = 0.0;
        for (std::size_t index = 0; index < static_cast<std::size_t>(table.row_count); ++index) {
            const bool mismatch = ((pattern >> index) & 1ULL) != 0;
            if (mismatch) {
                no_delta -= before_weights[index];
                toggle_delta -= before_weights[index];
                no_delta += after_weights[index];
            } else {
                toggle_delta += after_weights[index];
            }
        }
        if (table.small_enabled) {
            table.no_delta_small[pattern] = no_delta;
            table.toggle_delta_small[pattern] = toggle_delta;
            table.no_delta_small_s1024[pattern] = quantize_metric(no_delta, 1024);
            table.toggle_delta_small_s1024[pattern] = quantize_metric(toggle_delta, 1024);
        } else {
            table.no_delta[pattern] = no_delta;
            table.toggle_delta[pattern] = toggle_delta;
            table.no_delta_s1024[pattern] = quantize_metric(no_delta, 1024);
            table.toggle_delta_s1024[pattern] = quantize_metric(toggle_delta, 1024);
        }
    }
    return table;
}

double local_pattern_no_delta(const LocalPatternTable& table, std::size_t pattern) {
    return table.small_enabled ? table.no_delta_small[pattern] : table.no_delta[pattern];
}

double local_pattern_toggle_delta(const LocalPatternTable& table, std::size_t pattern) {
    return table.small_enabled ? table.toggle_delta_small[pattern] : table.toggle_delta[pattern];
}

std::int64_t local_pattern_no_delta_int(
    const LocalPatternTable& table,
    std::size_t pattern,
    int int_metric_scale
) {
    if (int_metric_scale == 1024) {
        return table.small_enabled ? table.no_delta_small_s1024[pattern] : table.no_delta_s1024[pattern];
    }
    return quantize_metric(local_pattern_no_delta(table, pattern), int_metric_scale);
}

std::int64_t local_pattern_toggle_delta_int(
    const LocalPatternTable& table,
    std::size_t pattern,
    int int_metric_scale
) {
    if (int_metric_scale == 1024) {
        return table.small_enabled ? table.toggle_delta_small_s1024[pattern] : table.toggle_delta_s1024[pattern];
    }
    return quantize_metric(local_pattern_toggle_delta(table, pattern), int_metric_scale);
}

std::int64_t row_term_parity_int(const RowTerm& term, int int_metric_scale) {
    return quantize_metric_s1024_cached(term.parity, term.parity_s1024, int_metric_scale);
}

std::int64_t column_no_error_log_const_int(const Column& column, int int_metric_scale) {
    return quantize_metric_s1024_cached(
        column.no_error_log_const,
        column.no_error_log_const_s1024,
        int_metric_scale
    );
}

std::int64_t column_toggle_logodds_int(const Column& column, int int_metric_scale) {
    return quantize_metric_s1024_cached(column.toggle_logodds, column.toggle_logodds_s1024, int_metric_scale);
}

std::unique_ptr<NativeModel> parse_model(PyObject* spec) {
    if (!PyDict_Check(spec)) {
        throw std::runtime_error("native model spec must be a dict");
    }
    auto model = std::make_unique<NativeModel>();
    model->num_detectors = static_cast<int>(as_long(get_item(spec, "num_detectors"), "num_detectors"));
    model->num_observables = static_cast<int>(as_long(get_item(spec, "num_observables"), "num_observables"));
    model->n_limbs = static_cast<int>(as_long(get_item(spec, "n_limbs"), "n_limbs"));
    model->n_logical_limbs = static_cast<int>(as_long(get_item(spec, "n_logical_limbs"), "n_logical_limbs"));
    PyObject* collect_phase_timing_obj = PyDict_GetItemString(spec, "collect_phase_timing");
    if (collect_phase_timing_obj != nullptr) {
        const int truth_value = PyObject_IsTrue(collect_phase_timing_obj);
        if (truth_value < 0) {
            throw std::runtime_error("invalid bool field: collect_phase_timing");
        }
        model->collect_phase_timing = truth_value != 0;
    }
    PyObject* force_full_key_obj = PyDict_GetItemString(spec, "force_full_key");
    if (force_full_key_obj != nullptr) {
        const int truth_value = PyObject_IsTrue(force_full_key_obj);
        if (truth_value < 0) {
            throw std::runtime_error("invalid bool field: force_full_key");
        }
        model->force_full_key = truth_value != 0;
    }
    if (model->num_detectors <= 0 || model->num_observables <= 0) {
        throw std::runtime_error("native model dimensions must be positive");
    }
    if (model->n_limbs <= 0 || model->n_limbs > kMaxLimbs) {
        throw std::runtime_error("native model limb count is unsupported");
    }
    if (model->n_logical_limbs <= 0 || model->n_logical_limbs > kMaxLogicalLimbs) {
        throw std::runtime_error("native model logical limb count is unsupported");
    }
    PyObject* columns_obj = get_item(spec, "columns");
    PyObject* columns_seq = PySequence_Fast(columns_obj, "columns must be a sequence");
    if (columns_seq == nullptr) {
        throw std::runtime_error("columns must be a sequence");
    }
    const Py_ssize_t column_count = PySequence_Fast_GET_SIZE(columns_seq);
    model->columns.reserve(static_cast<std::size_t>(column_count));
    PyObject** items = PySequence_Fast_ITEMS(columns_seq);
    std::array<std::uint64_t, kMaxLimbs> previous_active{};
    for (Py_ssize_t column_index = 0; column_index < column_count; ++column_index) {
        PyObject* column_dict = items[column_index];
        if (!PyDict_Check(column_dict)) {
            Py_DECREF(columns_seq);
            throw std::runtime_error("native column spec must be a dict");
        }
        Column column;
        column.no_error_log_const = as_double(get_item(column_dict, "no_error_log_const"), "no_error_log_const");
        column.toggle_logodds = as_double(get_item(column_dict, "toggle_logodds"), "toggle_logodds");
        column.no_error_log_const_s1024 = quantize_metric(column.no_error_log_const, 1024);
        column.toggle_logodds_s1024 = quantize_metric(column.toggle_logodds, 1024);
        column.toggle_logical =
            parse_logical_limbs(get_item(column_dict, "toggle_logical_limbs"), model->n_logical_limbs, "toggle_logical_limbs");
        column.toggle_detector = parse_limbs(get_item(column_dict, "toggle_detector_limbs"), model->n_limbs, "toggle_detector_limbs");
        column.close_mask = parse_limbs(get_item(column_dict, "close_limbs"), model->n_limbs, "close_limbs");
        column.active_mask = parse_limbs(get_item(column_dict, "active_limbs"), model->n_limbs, "active_limbs");
        column.before_slot_by_limb4.fill(-1);
        column.active_before_slots4.fill(-1);
        for (int limb = 0; limb < model->n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            column.active_toggle_detector[limb_index] =
                column.toggle_detector[limb_index] & column.active_mask[limb_index];
        }
        column.no_child_injective4 = true;
        column.toggle_has_new_active_bit = false;
        for (int limb = 0; limb < model->n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            if ((previous_active[limb_index] & ~column.active_mask[limb_index]) != 0) {
                column.no_child_injective4 = false;
            }
            if ((column.active_toggle_detector[limb_index] & ~previous_active[limb_index]) != 0) {
                column.toggle_has_new_active_bit = true;
            }
        }
        column.before_terms = parse_row_terms(column_dict, "before");
        column.after_terms = parse_row_terms(column_dict, "after");
        column.before_active_count4 = 0;
        for (int limb = 0; limb < model->n_limbs; ++limb) {
            if (previous_active[static_cast<std::size_t>(limb)] != 0) {
                if (column.before_active_count4 < kCompactLimbs) {
                    column.before_active_limbs4[static_cast<std::size_t>(column.before_active_count4)] = limb;
                    column.before_slot_by_limb4[static_cast<std::size_t>(limb)] = column.before_active_count4;
                }
                column.before_active_count4 += 1;
            }
        }
        column.close_nonzero_count = fill_nonzero_limbs(
            column.close_mask,
            model->n_limbs,
            column.close_nonzero_limbs
        );
        column.active_nonzero_count = fill_nonzero_limbs(
            column.active_mask,
            model->n_limbs,
            column.active_nonzero_limbs
        );
        column.toggle_nonzero_count = fill_nonzero_limbs(
            column.active_toggle_detector,
            model->n_limbs,
            column.toggle_nonzero_limbs
        );
        for (int slot = 0; slot < column.active_nonzero_count && slot < kCompactLimbs; ++slot) {
            const int limb = column.active_nonzero_limbs[static_cast<std::size_t>(slot)];
            column.active_before_slots4[static_cast<std::size_t>(slot)] =
                column.before_slot_by_limb4[static_cast<std::size_t>(limb)];
            column.active_slot_masks4[static_cast<std::size_t>(slot)] =
                column.active_mask[static_cast<std::size_t>(limb)];
            column.active_toggle_slots4[static_cast<std::size_t>(slot)] =
                column.active_toggle_detector[static_cast<std::size_t>(limb)];
        }
        column.active4_no_child_identity = column.close_nonzero_count == 0
            && column.before_active_count4 == column.active_nonzero_count
            && column.active_nonzero_count <= kCompactLimbs;
        if (column.active4_no_child_identity) {
            for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
                const std::size_t slot_index = static_cast<std::size_t>(slot);
                const int before_slot = column.active_before_slots4[slot_index];
                const int limb = column.active_nonzero_limbs[slot_index];
                if (
                    before_slot != slot
                    || ((previous_active[static_cast<std::size_t>(limb)]
                         & ~column.active_slot_masks4[slot_index]) != 0)
                ) {
                    column.active4_no_child_identity = false;
                    break;
                }
            }
        }
        if (column.before_active_count4 > kCompactLimbs || column.active_nonzero_count > kCompactLimbs) {
            model->active4_supported = false;
        }
        for (RowTerm& term : column.before_terms) {
            term.active4_slot = column.before_slot_by_limb4[static_cast<std::size_t>(term.limb)];
        }
        for (RowTerm& term : column.after_terms) {
            term.active4_slot = column.before_slot_by_limb4[static_cast<std::size_t>(term.limb)];
        }
        column.pattern_table = compile_local_pattern_table(column.before_terms, column.after_terms);
        for (int index = 0; index < column.pattern_table.row_count; ++index) {
            RowTerm& term = column.pattern_table.rows[static_cast<std::size_t>(index)];
            term.active4_slot = column.before_slot_by_limb4[static_cast<std::size_t>(term.limb)];
        }
        column.sparse_clear_supported = true;
        for (int limb = 0; limb < model->n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            const std::uint64_t cleared_by_active =
                previous_active[limb_index] & ~column.active_mask[limb_index];
            if (cleared_by_active != column.close_mask[limb_index]) {
                column.sparse_clear_supported = false;
                break;
            }
        }
        previous_active = column.active_mask;
        model->columns.push_back(std::move(column));
    }
    Py_DECREF(columns_seq);
    return model;
}

std::unique_ptr<ChoiceNativeModel> parse_choice_model(PyObject* spec) {
    if (!PyDict_Check(spec)) {
        throw std::runtime_error("native choice model spec must be a dict");
    }
    auto model = std::make_unique<ChoiceNativeModel>();
    model->num_detectors = static_cast<int>(as_long(get_item(spec, "num_detectors"), "num_detectors"));
    model->num_observables = static_cast<int>(as_long(get_item(spec, "num_observables"), "num_observables"));
    model->n_limbs = static_cast<int>(as_long(get_item(spec, "n_limbs"), "n_limbs"));
    model->n_logical_limbs = static_cast<int>(as_long(get_item(spec, "n_logical_limbs"), "n_logical_limbs"));
    PyObject* collect_phase_timing_obj = PyDict_GetItemString(spec, "collect_phase_timing");
    if (collect_phase_timing_obj != nullptr) {
        const int truth_value = PyObject_IsTrue(collect_phase_timing_obj);
        if (truth_value < 0) {
            throw std::runtime_error("invalid bool field: collect_phase_timing");
        }
        model->collect_phase_timing = truth_value != 0;
    }
    if (model->num_detectors <= 0 || model->num_observables <= 0) {
        throw std::runtime_error("native choice model dimensions must be positive");
    }
    if (model->n_limbs <= 0 || model->n_limbs > kMaxLimbs) {
        throw std::runtime_error("native choice model limb count is unsupported");
    }
    if (model->n_logical_limbs <= 0 || model->n_logical_limbs > kMaxLogicalLimbs) {
        throw std::runtime_error("native choice model logical limb count is unsupported");
    }

    PyObject* columns_obj = get_item(spec, "columns");
    PyObject* columns_seq = PySequence_Fast(columns_obj, "choice columns must be a sequence");
    if (columns_seq == nullptr) {
        throw std::runtime_error("choice columns must be a sequence");
    }
    const Py_ssize_t column_count = PySequence_Fast_GET_SIZE(columns_seq);
    model->columns.reserve(static_cast<std::size_t>(column_count));
    PyObject** column_items = PySequence_Fast_ITEMS(columns_seq);
    for (Py_ssize_t column_index = 0; column_index < column_count; ++column_index) {
        PyObject* column_dict = column_items[column_index];
        if (!PyDict_Check(column_dict)) {
            Py_DECREF(columns_seq);
            throw std::runtime_error("native choice column spec must be a dict");
        }
        ChoiceColumn column;
        column.close_mask = parse_limbs(get_item(column_dict, "close_limbs"), model->n_limbs, "close_limbs");
        column.active_mask = parse_limbs(get_item(column_dict, "active_limbs"), model->n_limbs, "active_limbs");
        column.close_nonzero_count = fill_nonzero_limbs(
            column.close_mask,
            model->n_limbs,
            column.close_nonzero_limbs
        );
        column.before_terms = parse_row_terms(column_dict, "before");
        column.after_terms = parse_row_terms(column_dict, "after");

        std::vector<double> log_priors = parse_double_list(get_item(column_dict, "log_priors"), "log_priors");
        PyObject* detector_obj = get_item(column_dict, "detector_limbs");
        PyObject* detector_seq = PySequence_Fast(detector_obj, "detector_limbs must be a sequence");
        if (detector_seq == nullptr) {
            Py_DECREF(columns_seq);
            throw std::runtime_error("detector_limbs must be a sequence");
        }
        PyObject* logical_obj = get_item(column_dict, "logical_masks");
        PyObject* logical_seq = PySequence_Fast(logical_obj, "logical_masks must be a sequence");
        if (logical_seq == nullptr) {
            Py_DECREF(detector_seq);
            Py_DECREF(columns_seq);
            throw std::runtime_error("logical_masks must be a sequence");
        }
        const Py_ssize_t option_count = PySequence_Fast_GET_SIZE(detector_seq);
        if (
            option_count != PySequence_Fast_GET_SIZE(logical_seq) ||
            option_count != static_cast<Py_ssize_t>(log_priors.size())
        ) {
            Py_DECREF(logical_seq);
            Py_DECREF(detector_seq);
            Py_DECREF(columns_seq);
            throw std::runtime_error("choice option arrays have mismatched lengths");
        }
        if (option_count <= 0) {
            Py_DECREF(logical_seq);
            Py_DECREF(detector_seq);
            Py_DECREF(columns_seq);
            throw std::runtime_error("choice column must contain at least one local option");
        }
        column.options.reserve(static_cast<std::size_t>(option_count));
        PyObject** detector_items = PySequence_Fast_ITEMS(detector_seq);
        PyObject** logical_items = PySequence_Fast_ITEMS(logical_seq);
        for (Py_ssize_t option_index = 0; option_index < option_count; ++option_index) {
            ChoiceOption option;
            option.log_prior = log_priors[static_cast<std::size_t>(option_index)];
            option.detector = parse_limbs(detector_items[option_index], model->n_limbs, "detector_limbs");
            option.logical = parse_logical_limbs(logical_items[option_index], model->n_logical_limbs, "logical_masks");
            column.options.push_back(option);
        }
        Py_DECREF(logical_seq);
        Py_DECREF(detector_seq);
        model->columns.push_back(std::move(column));
    }
    Py_DECREF(columns_seq);
    return model;
}

void capsule_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, "frontier_native_model");
    if (ptr != nullptr) {
        delete static_cast<NativeModel*>(ptr);
    } else {
        PyErr_Clear();
    }
}

void choice_capsule_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, "frontier_native_choice_model");
    if (ptr != nullptr) {
        delete static_cast<ChoiceNativeModel*>(ptr);
    } else {
        PyErr_Clear();
    }
}

NativeModel* model_from_capsule(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, "frontier_native_model");
    if (ptr == nullptr) {
        throw std::runtime_error("invalid native frontier model capsule");
    }
    return static_cast<NativeModel*>(ptr);
}

ChoiceNativeModel* choice_model_from_capsule(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, "frontier_native_choice_model");
    if (ptr == nullptr) {
        throw std::runtime_error("invalid native frontier choice model capsule");
    }
    return static_cast<ChoiceNativeModel*>(ptr);
}

std::array<std::uint64_t, kMaxLimbs> parse_syndrome_limbs(PyObject* obj, int n_limbs) {
    std::array<std::uint64_t, kMaxLimbs> out{};
    PyObject* seq = PySequence_Fast(obj, "syndrome_limbs");
    if (seq == nullptr) {
        throw std::runtime_error("invalid syndrome limb sequence");
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (static_cast<int>(size) != n_limbs) {
        Py_DECREF(seq);
        throw std::runtime_error("syndrome limb count does not match native model");
    }
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (int index = 0; index < n_limbs; ++index) {
        out[static_cast<std::size_t>(index)] =
            static_cast<std::uint64_t>(as_ull(items[index], "syndrome_limbs"));
    }
    Py_DECREF(seq);
    return out;
}

std::vector<std::array<std::uint64_t, kMaxLimbs>> parse_many_syndrome_limbs(PyObject* obj, int n_limbs) {
    PyObject* seq = PySequence_Fast(obj, "syndrome limb batch must be a sequence");
    if (seq == nullptr) {
        throw std::runtime_error("syndrome limb batch must be a sequence");
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    std::vector<std::array<std::uint64_t, kMaxLimbs>> out;
    out.reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t index = 0; index < size; ++index) {
        out.push_back(parse_syndrome_limbs(items[index], n_limbs));
    }
    Py_DECREF(seq);
    return out;
}

int popcount_limbs(const std::array<std::uint64_t, kMaxLimbs>& limbs, int n_limbs) {
    int count = 0;
    for (int limb = 0; limb < n_limbs; ++limb) {
        count += __builtin_popcountll(limbs[static_cast<std::size_t>(limb)]);
    }
    return count;
}

bool limb_overlap_nonzero(
    const std::array<std::uint64_t, kMaxLimbs>& lhs,
    const std::array<std::uint64_t, kMaxLimbs>& rhs,
    int n_limbs
) {
    for (int limb = 0; limb < n_limbs; ++limb) {
        if ((lhs[static_cast<std::size_t>(limb)] & rhs[static_cast<std::size_t>(limb)]) != 0) {
            return true;
        }
    }
    return false;
}

void set_limb_bit(std::array<std::uint64_t, kMaxLimbs>& limbs, int row) {
    limbs[static_cast<std::size_t>(row / 64)] |= (1ULL << static_cast<unsigned>(row % 64));
}

bool get_limb_bit(const std::array<std::uint64_t, kMaxLimbs>& limbs, int row) {
    return ((limbs[static_cast<std::size_t>(row / 64)] >> static_cast<unsigned>(row % 64)) & 1ULL) != 0;
}

double binary_toggle_probability(const Column& column) {
    if (!std::isfinite(column.no_error_log_const) || !std::isfinite(column.toggle_logodds)) {
        return 0.0;
    }
    const double log_p1 = column.no_error_log_const + column.toggle_logodds;
    if (log_p1 <= std::log(std::numeric_limits<double>::min())) {
        return 0.0;
    }
    const double p1 = std::exp(log_p1);
    if (!std::isfinite(p1)) return 1.0;
    return std::min(1.0, std::max(0.0, p1));
}

void finalize_native_model_columns(NativeModel& model) {
    model.active4_supported = true;
    std::array<std::uint64_t, kMaxLimbs> previous_active{};
    for (Column& column : model.columns) {
        column.before_slot_by_limb4.fill(-1);
        column.active_before_slots4.fill(-1);
        for (int limb = 0; limb < model.n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            column.active_toggle_detector[limb_index] =
                column.toggle_detector[limb_index] & column.active_mask[limb_index];
        }
        column.no_child_injective4 = true;
        column.toggle_has_new_active_bit = false;
        for (int limb = 0; limb < model.n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            if ((previous_active[limb_index] & ~column.active_mask[limb_index]) != 0) {
                column.no_child_injective4 = false;
            }
            if ((column.active_toggle_detector[limb_index] & ~previous_active[limb_index]) != 0) {
                column.toggle_has_new_active_bit = true;
            }
        }
        column.before_active_count4 = 0;
        for (int limb = 0; limb < model.n_limbs; ++limb) {
            if (previous_active[static_cast<std::size_t>(limb)] != 0) {
                if (column.before_active_count4 < kCompactLimbs) {
                    column.before_active_limbs4[static_cast<std::size_t>(column.before_active_count4)] = limb;
                    column.before_slot_by_limb4[static_cast<std::size_t>(limb)] = column.before_active_count4;
                }
                column.before_active_count4 += 1;
            }
        }
        column.close_nonzero_count = fill_nonzero_limbs(
            column.close_mask,
            model.n_limbs,
            column.close_nonzero_limbs
        );
        column.active_nonzero_count = fill_nonzero_limbs(
            column.active_mask,
            model.n_limbs,
            column.active_nonzero_limbs
        );
        column.toggle_nonzero_count = fill_nonzero_limbs(
            column.active_toggle_detector,
            model.n_limbs,
            column.toggle_nonzero_limbs
        );
        for (int slot = 0; slot < column.active_nonzero_count && slot < kCompactLimbs; ++slot) {
            const int limb = column.active_nonzero_limbs[static_cast<std::size_t>(slot)];
            column.active_before_slots4[static_cast<std::size_t>(slot)] =
                column.before_slot_by_limb4[static_cast<std::size_t>(limb)];
            column.active_slot_masks4[static_cast<std::size_t>(slot)] =
                column.active_mask[static_cast<std::size_t>(limb)];
            column.active_toggle_slots4[static_cast<std::size_t>(slot)] =
                column.active_toggle_detector[static_cast<std::size_t>(limb)];
        }
        column.active4_no_child_identity = column.close_nonzero_count == 0
            && column.before_active_count4 == column.active_nonzero_count
            && column.active_nonzero_count <= kCompactLimbs;
        if (column.active4_no_child_identity) {
            for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
                const std::size_t slot_index = static_cast<std::size_t>(slot);
                const int before_slot = column.active_before_slots4[slot_index];
                const int limb = column.active_nonzero_limbs[slot_index];
                if (
                    before_slot != slot
                    || ((previous_active[static_cast<std::size_t>(limb)]
                         & ~column.active_slot_masks4[slot_index]) != 0)
                ) {
                    column.active4_no_child_identity = false;
                    break;
                }
            }
        }
        if (column.before_active_count4 > kCompactLimbs || column.active_nonzero_count > kCompactLimbs) {
            model.active4_supported = false;
        }
        for (RowTerm& term : column.before_terms) {
            term.active4_slot = column.before_slot_by_limb4[static_cast<std::size_t>(term.limb)];
            term.parity_s1024 = quantize_metric(term.parity, 1024);
        }
        for (RowTerm& term : column.after_terms) {
            term.active4_slot = column.before_slot_by_limb4[static_cast<std::size_t>(term.limb)];
            term.parity_s1024 = quantize_metric(term.parity, 1024);
        }
        column.pattern_table = compile_local_pattern_table(column.before_terms, column.after_terms);
        for (int index = 0; index < column.pattern_table.row_count; ++index) {
            RowTerm& term = column.pattern_table.rows[static_cast<std::size_t>(index)];
            term.active4_slot = column.before_slot_by_limb4[static_cast<std::size_t>(term.limb)];
        }
        column.sparse_clear_supported = true;
        for (int limb = 0; limb < model.n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            const std::uint64_t cleared_by_active =
                previous_active[limb_index] & ~column.active_mask[limb_index];
            if (cleared_by_active != column.close_mask[limb_index]) {
                column.sparse_clear_supported = false;
                break;
            }
        }
        previous_active = column.active_mask;
    }
}

struct OverlapFirstStageModel {
    NativeModel model;
    int candidate_cols = 0;
    int reduced_rows = 0;
    int active_syndrome_weight = 0;
    int uncovered_active_rows = 0;
};

OverlapFirstStageModel build_overlap1_first_stage_model(
    const NativeModel& source,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    OverlapFirstStageModel out;
    out.model.num_detectors = source.num_detectors;
    out.model.num_observables = source.num_observables;
    out.model.n_limbs = source.n_limbs;
    out.model.n_logical_limbs = source.n_logical_limbs;
    out.model.collect_phase_timing = source.collect_phase_timing;
    out.model.force_full_key = source.force_full_key;
    out.active_syndrome_weight = popcount_limbs(syndrome, source.n_limbs);

    std::vector<int> selected;
    selected.reserve(source.columns.size());
    for (std::size_t index = 0; index < source.columns.size(); ++index) {
        if (limb_overlap_nonzero(source.columns[index].toggle_detector, syndrome, source.n_limbs)) {
            selected.push_back(static_cast<int>(index));
        }
    }
    out.candidate_cols = static_cast<int>(selected.size());
    out.model.columns.reserve(selected.size());
    for (int source_index : selected) {
        const Column& src = source.columns[static_cast<std::size_t>(source_index)];
        Column dst;
        dst.no_error_log_const = src.no_error_log_const;
        dst.toggle_logodds = src.toggle_logodds;
        dst.no_error_log_const_s1024 = src.no_error_log_const_s1024;
        dst.toggle_logodds_s1024 = src.toggle_logodds_s1024;
        dst.toggle_detector = src.toggle_detector;
        dst.toggle_logical = src.toggle_logical;
        out.model.columns.push_back(std::move(dst));
    }

    const int n_columns = static_cast<int>(out.model.columns.size());
    std::vector<std::vector<int>> row_touches(static_cast<std::size_t>(source.num_detectors));
    std::array<std::uint64_t, kMaxLimbs> touched_mask{};
    for (int local_index = 0; local_index < n_columns; ++local_index) {
        const Column& column = out.model.columns[static_cast<std::size_t>(local_index)];
        for (int limb = 0; limb < source.n_limbs; ++limb) {
            std::uint64_t bits = column.toggle_detector[static_cast<std::size_t>(limb)];
            while (bits != 0) {
                const int bit = __builtin_ctzll(bits);
                const int row = limb * 64 + bit;
                if (row < source.num_detectors) {
                    row_touches[static_cast<std::size_t>(row)].push_back(local_index);
                    set_limb_bit(touched_mask, row);
                }
                bits &= bits - 1;
            }
        }
    }
    std::array<std::uint64_t, kMaxLimbs> reduced_mask = touched_mask;
    for (int row = 0; row < source.num_detectors; ++row) {
        if (get_limb_bit(syndrome, row)) {
            set_limb_bit(reduced_mask, row);
            if (row_touches[static_cast<std::size_t>(row)].empty()) {
                out.uncovered_active_rows += 1;
            }
        }
    }
    out.reduced_rows = popcount_limbs(reduced_mask, source.n_limbs);

    if (n_columns == 0) {
        return out;
    }

    std::vector<std::vector<int>> rows_start(static_cast<std::size_t>(n_columns));
    std::vector<std::vector<int>> rows_end(static_cast<std::size_t>(n_columns));
    for (int row = 0; row < source.num_detectors; ++row) {
        const std::vector<int>& touches = row_touches[static_cast<std::size_t>(row)];
        if (touches.empty()) continue;
        rows_start[static_cast<std::size_t>(touches.front())].push_back(row);
        rows_end[static_cast<std::size_t>(touches.back())].push_back(row);
    }

    std::array<std::uint64_t, kMaxLimbs> active_mask{};
    for (int column_index = 0; column_index < n_columns; ++column_index) {
        Column& column = out.model.columns[static_cast<std::size_t>(column_index)];
        for (int row : rows_start[static_cast<std::size_t>(column_index)]) {
            const std::vector<int>& touches = row_touches[static_cast<std::size_t>(row)];
            if (!touches.empty() && touches.back() > column_index) {
                set_limb_bit(active_mask, row);
            }
        }
        for (int row : rows_end[static_cast<std::size_t>(column_index)]) {
            set_limb_bit(column.close_mask, row);
            active_mask[static_cast<std::size_t>(row / 64)] &= ~(1ULL << static_cast<unsigned>(row % 64));
        }
        column.active_mask = active_mask;
    }

    const double tiny = std::numeric_limits<double>::min();
    for (int row = 0; row < source.num_detectors; ++row) {
        const std::vector<int>& touches = row_touches[static_cast<std::size_t>(row)];
        if (touches.empty()) continue;
        const int total = static_cast<int>(touches.size());
        std::vector<double> suffix_even(static_cast<std::size_t>(total + 1), 0.0);
        std::vector<double> suffix_odd(static_cast<std::size_t>(total + 1), 0.0);
        suffix_even[static_cast<std::size_t>(total)] = 1.0;
        for (int offset = total - 1; offset >= 0; --offset) {
            const Column& column = out.model.columns[static_cast<std::size_t>(touches[static_cast<std::size_t>(offset)])];
            const double flip_prob = binary_toggle_probability(column);
            suffix_even[static_cast<std::size_t>(offset)] =
                (1.0 - flip_prob) * suffix_even[static_cast<std::size_t>(offset + 1)]
                + flip_prob * suffix_odd[static_cast<std::size_t>(offset + 1)];
            suffix_odd[static_cast<std::size_t>(offset)] =
                (1.0 - flip_prob) * suffix_odd[static_cast<std::size_t>(offset + 1)]
                + flip_prob * suffix_even[static_cast<std::size_t>(offset + 1)];
        }
        for (int offset = 0; offset < total; ++offset) {
            const int column_index = touches[static_cast<std::size_t>(offset)];
            Column& column = out.model.columns[static_cast<std::size_t>(column_index)];
            const int limb = row / 64;
            const std::uint64_t bit = 1ULL << static_cast<unsigned>(row % 64);
            if (offset > 0) {
                RowTerm term;
                term.limb = limb;
                term.bit = bit;
                term.parity =
                    std::log(std::max(suffix_odd[static_cast<std::size_t>(offset)], tiny))
                    - std::log(std::max(suffix_even[static_cast<std::size_t>(offset)], tiny));
                term.parity_s1024 = quantize_metric(term.parity, 1024);
                column.before_terms.push_back(term);
            }
            if (offset + 1 < total) {
                RowTerm term;
                term.limb = limb;
                term.bit = bit;
                term.parity =
                    std::log(std::max(suffix_odd[static_cast<std::size_t>(offset + 1)], tiny))
                    - std::log(std::max(suffix_even[static_cast<std::size_t>(offset + 1)], tiny));
                term.parity_s1024 = quantize_metric(term.parity, 1024);
                column.after_terms.push_back(term);
            }
        }
    }

    finalize_native_model_columns(out.model);
    return out;
}

struct FirstStageDecodePayload {
    DecodeResult result;
    std::string status_override;
    int candidate_cols = 0;
    int reduced_rows = 0;
    int active_syndrome_weight = 0;
    int uncovered_active_rows = 0;
    double build_time_s = 0.0;
    double decode_time_s = 0.0;
};

struct Stage1NocapStage2Payload {
    DecodeResult selected;
    DecodeResult final_forward;
    DecodeResult final_backward;
    bool selected_forward = true;
    bool used_stage2 = false;
    std::string stage1_status;
    FirstStageDecodePayload stage1_forward;
    FirstStageDecodePayload stage1_backward;
    bool stage1_selected_forward = true;
    bool stage1_agree = false;
};

std::pair<bool, bool> closure_acceptance_sparse(
    const Key& parent_key,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    bool toggle_finite
) {
    bool no_accepted = true;
    bool toggle_accepted = toggle_finite;
    for (int index = 0; index < column.close_nonzero_count; ++index) {
        const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
        const std::size_t limb_index = static_cast<std::size_t>(limb);
        const std::uint64_t close_mask = column.close_mask[limb_index];
        const std::uint64_t parent_mismatch = (parent_key.det[limb_index] ^ syndrome[limb_index]) & close_mask;
        if (parent_mismatch != 0) {
            no_accepted = false;
        }
        if (toggle_finite) {
            const std::uint64_t toggle_mismatch =
                parent_mismatch ^ (column.toggle_detector[limb_index] & close_mask);
            if (toggle_mismatch != 0) {
                toggle_accepted = false;
            }
        }
        if (!no_accepted && (!toggle_finite || !toggle_accepted)) {
            break;
        }
    }
    return {no_accepted, toggle_accepted};
}

double parent_mismatch_row_value(
    const State& parent,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const RowTerm& term
) {
    const std::uint64_t bit =
        (parent.key.det[static_cast<std::size_t>(term.limb)] ^ syndrome[static_cast<std::size_t>(term.limb)])
        & term.bit;
    return bit != 0 ? 1.0 : 0.0;
}

std::size_t local_pattern_bit(
    const Key& key,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const RowTerm& term,
    std::size_t index
) {
    const std::uint64_t bit =
        (key.det[static_cast<std::size_t>(term.limb)] ^ syndrome[static_cast<std::size_t>(term.limb)])
        & term.bit;
    return bit != 0 ? (static_cast<std::size_t>(1) << index) : 0U;
}

std::size_t local_pattern_index(
    const Key& key,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const LocalPatternTable& table
) {
    if (table.small_enabled) {
        std::size_t pattern = 0;
        switch (table.row_count) {
            case 6:
                pattern |= local_pattern_bit(key, syndrome, table.rows[5], 5);
                [[fallthrough]];
            case 5:
                pattern |= local_pattern_bit(key, syndrome, table.rows[4], 4);
                [[fallthrough]];
            case 4:
                pattern |= local_pattern_bit(key, syndrome, table.rows[3], 3);
                [[fallthrough]];
            case 3:
                pattern |= local_pattern_bit(key, syndrome, table.rows[2], 2);
                [[fallthrough]];
            case 2:
                pattern |= local_pattern_bit(key, syndrome, table.rows[1], 1);
                [[fallthrough]];
            case 1:
                pattern |= local_pattern_bit(key, syndrome, table.rows[0], 0);
                break;
            default:
                break;
        }
        return pattern;
    }
    std::size_t pattern = 0;
    for (std::size_t index = 0; index < static_cast<std::size_t>(table.row_count); ++index) {
        pattern |= local_pattern_bit(key, syndrome, table.rows[index], index);
    }
    return pattern;
}

std::pair<double, double> child_parities(
    const State& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    double base = parent.parity_logodds;
    for (const RowTerm& term : column.before_terms) {
        if (parent_mismatch_row_value(parent, syndrome, term) != 0.0) {
            base -= term.parity;
        }
    }
    double no_parity = base;
    double toggle_parity = base;
    for (const RowTerm& term : column.after_terms) {
        if (parent_mismatch_row_value(parent, syndrome, term) != 0.0) {
            no_parity += term.parity;
        } else {
            toggle_parity += term.parity;
        }
    }
    return {no_parity, toggle_parity};
}

std::pair<double, double> child_parities_fast(
    const State& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    if (column.pattern_table.enabled) {
        const std::size_t pattern = local_pattern_index(parent.key, syndrome, column.pattern_table);
        return {
            parent.parity_logodds + local_pattern_no_delta(column.pattern_table, pattern),
            parent.parity_logodds + local_pattern_toggle_delta(column.pattern_table, pattern),
        };
    }
    return child_parities(parent, column, syndrome);
}

struct StateInt {
    Key key;
    std::int64_t logmass = kIntMetricNegInf;
    std::int64_t parity_logodds = 0;
};

std::pair<std::int64_t, std::int64_t> child_parities_int(
    const StateInt& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int int_metric_scale
) {
    if (column.pattern_table.enabled) {
        const std::size_t pattern = local_pattern_index(parent.key, syndrome, column.pattern_table);
        return {
            parent.parity_logodds + local_pattern_no_delta_int(column.pattern_table, pattern, int_metric_scale),
            parent.parity_logodds + local_pattern_toggle_delta_int(column.pattern_table, pattern, int_metric_scale),
        };
    }

    std::int64_t base = parent.parity_logodds;
    for (const RowTerm& term : column.before_terms) {
        const std::uint64_t bit =
            (parent.key.det[static_cast<std::size_t>(term.limb)] ^ syndrome[static_cast<std::size_t>(term.limb)])
            & term.bit;
        if (bit != 0) {
            base -= row_term_parity_int(term, int_metric_scale);
        }
    }
    std::int64_t no_parity = base;
    std::int64_t toggle_parity = base;
    for (const RowTerm& term : column.after_terms) {
        const std::int64_t parity = row_term_parity_int(term, int_metric_scale);
        const std::uint64_t bit =
            (parent.key.det[static_cast<std::size_t>(term.limb)] ^ syndrome[static_cast<std::size_t>(term.limb)])
            & term.bit;
        if (bit != 0) {
            no_parity += parity;
        } else {
            toggle_parity += parity;
        }
    }
    return {no_parity, toggle_parity};
}

void fill_no_child_key(Key& child, const Key& parent_key, const Column& column, int n_limbs, const LogicalMask& logical) {
    child = parent_key;
    child.logical = logical;
    if (column.sparse_clear_supported) {
        for (int index = 0; index < column.close_nonzero_count; ++index) {
            const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            child.det[limb_index] &= column.active_mask[limb_index];
        }
    } else {
        for (int limb = 0; limb < n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            child.det[limb_index] &= column.active_mask[limb_index];
        }
    }
}

void apply_active_toggle_to_child(Key& child, const Column& column) {
    for (int index = 0; index < column.toggle_nonzero_count; ++index) {
        const int limb = column.toggle_nonzero_limbs[static_cast<std::size_t>(index)];
        const std::size_t limb_index = static_cast<std::size_t>(limb);
        child.det[limb_index] ^= column.active_toggle_detector[limb_index];
    }
}

void fill_toggle_child_key(Key& child, const Key& parent_key, const Column& column, int n_limbs, const LogicalMask& logical) {
    fill_no_child_key(child, parent_key, column, n_limbs, logical);
    apply_active_toggle_to_child(child, column);
}

struct CandidateBetter {
    int n_limbs = 0;

    bool operator()(const Candidate& lhs, const Candidate& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.logmass != rhs.logmass) return lhs.logmass > rhs.logmass;
        return KeyLess{n_limbs}(lhs.key, rhs.key);
    }
};

struct CandidateInt {
    Key key;
    std::int64_t logmass = kIntMetricNegInf;
    std::int64_t parity_logodds = 0;
    std::int64_t score = kIntMetricNegInf;
};

struct CandidateIntBetter {
    int n_limbs = 0;

    bool operator()(const CandidateInt& lhs, const CandidateInt& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.logmass != rhs.logmass) return lhs.logmass > rhs.logmass;
        return KeyLess{n_limbs}(lhs.key, rhs.key);
    }
};

void merge_candidate_maxlog(
    std::map<Key, CandidateInt, KeyLess>& merged_by_key,
    const CandidateInt& candidate
) {
    auto inserted = merged_by_key.emplace(candidate.key, candidate);
    if (!inserted.second && candidate.logmass > inserted.first->second.logmass) {
        inserted.first->second.logmass = candidate.logmass;
        inserted.first->second.parity_logodds = candidate.parity_logodds;
    }
}

struct Key4 {
    std::array<std::uint64_t, kCompactLimbs> det;
    LogicalMask logical{};
};

struct State4 {
    Key4 key;
    double logmass = -std::numeric_limits<double>::infinity();
    double parity_logodds = 0.0;
};

struct State4Int {
    Key4 key;
    std::int64_t logmass = kIntMetricNegInf;
    std::int64_t parity_logodds = 0;
};

struct Candidate4 {
    Key4 key;
    double logmass = -std::numeric_limits<double>::infinity();
    double parity_logodds = 0.0;
    double score = -std::numeric_limits<double>::infinity();
};

struct Candidate4Int {
    Key4 key;
    std::int64_t logmass = kIntMetricNegInf;
    std::int64_t parity_logodds = 0;
    std::int64_t score = kIntMetricNegInf;
};

bool key4_less(const Key4& lhs, const Key4& rhs, int n_limbs) {
    if (lhs.logical != rhs.logical) return lhs.logical < rhs.logical;
    for (int index = n_limbs - 1; index >= 0; --index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        if (lhs.det[limb_index] != rhs.det[limb_index]) {
            return lhs.det[limb_index] < rhs.det[limb_index];
        }
    }
    return false;
}

bool key4_equal(const Key4& lhs, const Key4& rhs, int n_limbs) {
    if (lhs.logical != rhs.logical) return false;
    for (int index = 0; index < n_limbs; ++index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        if (lhs.det[limb_index] != rhs.det[limb_index]) return false;
    }
    return true;
}

bool key4_equal_for_column(const Key4& lhs, const Key4& rhs, const Column& column, int n_limbs) {
    if (lhs.logical != rhs.logical) return false;
    if (column.sparse_clear_supported) {
        for (int offset = 0; offset < column.active_nonzero_count; ++offset) {
            const int index = column.active_nonzero_limbs[static_cast<std::size_t>(offset)];
            if (lhs.det[static_cast<std::size_t>(index)] != rhs.det[static_cast<std::size_t>(index)]) {
                return false;
            }
        }
        return true;
    }
    for (int index = 0; index < n_limbs; ++index) {
        const std::size_t limb_index = static_cast<std::size_t>(index);
        if (lhs.det[limb_index] != rhs.det[limb_index]) return false;
    }
    return true;
}

std::size_t hash_key4_for_column(const Key4& key, const Column& column, int n_limbs, int n_logical_limbs) {
    std::uint64_t h = static_cast<std::uint64_t>(hash_logical(key.logical, n_logical_limbs));
    if (column.sparse_clear_supported) {
        for (int offset = 0; offset < column.active_nonzero_count; ++offset) {
            const int index = column.active_nonzero_limbs[static_cast<std::size_t>(offset)];
            const std::uint64_t value = mix_hash_word(key.det[static_cast<std::size_t>(index)]);
            h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<std::size_t>(h);
    }
    for (int index = 0; index < n_limbs; ++index) {
        const std::uint64_t value = mix_hash_word(key.det[static_cast<std::size_t>(index)]);
        h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<std::size_t>(h);
}

bool det4_zero(const Key4& key, int n_limbs) {
    for (int index = 0; index < n_limbs; ++index) {
        if (key.det[static_cast<std::size_t>(index)] != 0) return false;
    }
    return true;
}

std::pair<bool, bool> closure_acceptance_sparse4(
    const Key4& parent_key,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    bool toggle_finite
) {
    bool no_accepted = true;
    bool toggle_accepted = toggle_finite;
    for (int index = 0; index < column.close_nonzero_count; ++index) {
        const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
        const std::size_t limb_index = static_cast<std::size_t>(limb);
        const std::uint64_t close_mask = column.close_mask[limb_index];
        const std::uint64_t parent_mismatch = (parent_key.det[limb_index] ^ syndrome[limb_index]) & close_mask;
        if (parent_mismatch != 0) {
            no_accepted = false;
        }
        if (toggle_finite) {
            const std::uint64_t toggle_mismatch =
                parent_mismatch ^ (column.toggle_detector[limb_index] & close_mask);
            if (toggle_mismatch != 0) {
                toggle_accepted = false;
            }
        }
        if (!no_accepted && (!toggle_finite || !toggle_accepted)) {
            break;
        }
    }
    return {no_accepted, toggle_accepted};
}

double parent_mismatch_row_value4(
    const State4& parent,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const RowTerm& term
) {
    const std::uint64_t bit =
        (parent.key.det[static_cast<std::size_t>(term.limb)] ^ syndrome[static_cast<std::size_t>(term.limb)])
        & term.bit;
    return bit != 0 ? 1.0 : 0.0;
}

std::size_t local_pattern_bit4(
    const Key4& key,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const RowTerm& term,
    std::size_t index
) {
    const std::uint64_t bit =
        (key.det[static_cast<std::size_t>(term.limb)] ^ syndrome[static_cast<std::size_t>(term.limb)])
        & term.bit;
    return bit != 0 ? (static_cast<std::size_t>(1) << index) : 0U;
}

std::size_t local_pattern_index4(
    const Key4& key,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const LocalPatternTable& table
) {
    if (table.small_enabled) {
        std::size_t pattern = 0;
        switch (table.row_count) {
            case 6:
                pattern |= local_pattern_bit4(key, syndrome, table.rows[5], 5);
                [[fallthrough]];
            case 5:
                pattern |= local_pattern_bit4(key, syndrome, table.rows[4], 4);
                [[fallthrough]];
            case 4:
                pattern |= local_pattern_bit4(key, syndrome, table.rows[3], 3);
                [[fallthrough]];
            case 3:
                pattern |= local_pattern_bit4(key, syndrome, table.rows[2], 2);
                [[fallthrough]];
            case 2:
                pattern |= local_pattern_bit4(key, syndrome, table.rows[1], 1);
                [[fallthrough]];
            case 1:
                pattern |= local_pattern_bit4(key, syndrome, table.rows[0], 0);
                break;
            default:
                break;
        }
        return pattern;
    }
    std::size_t pattern = 0;
    for (std::size_t index = 0; index < static_cast<std::size_t>(table.row_count); ++index) {
        pattern |= local_pattern_bit4(key, syndrome, table.rows[index], index);
    }
    return pattern;
}

std::pair<double, double> child_parities4(
    const State4& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    double base = parent.parity_logodds;
    for (const RowTerm& term : column.before_terms) {
        if (parent_mismatch_row_value4(parent, syndrome, term) != 0.0) {
            base -= term.parity;
        }
    }
    double no_parity = base;
    double toggle_parity = base;
    for (const RowTerm& term : column.after_terms) {
        if (parent_mismatch_row_value4(parent, syndrome, term) != 0.0) {
            no_parity += term.parity;
        } else {
            toggle_parity += term.parity;
        }
    }
    return {no_parity, toggle_parity};
}

std::pair<double, double> child_parities_fast4(
    const State4& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    if (column.pattern_table.enabled) {
        const std::size_t pattern = local_pattern_index4(parent.key, syndrome, column.pattern_table);
        return {
            parent.parity_logodds + local_pattern_no_delta(column.pattern_table, pattern),
            parent.parity_logodds + local_pattern_toggle_delta(column.pattern_table, pattern),
        };
    }
    return child_parities4(parent, column, syndrome);
}

void fill_no_child_key4(Key4& child, const Key4& parent_key, const Column& column, int n_limbs, const LogicalMask& logical) {
    child = parent_key;
    child.logical = logical;
    if (column.sparse_clear_supported) {
        for (int index = 0; index < column.close_nonzero_count; ++index) {
            const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            child.det[limb_index] &= column.active_mask[limb_index];
        }
    } else {
        for (int limb = 0; limb < n_limbs; ++limb) {
            const std::size_t limb_index = static_cast<std::size_t>(limb);
            child.det[limb_index] &= column.active_mask[limb_index];
        }
    }
}

void apply_active_toggle_to_child4(Key4& child, const Column& column) {
    for (int index = 0; index < column.toggle_nonzero_count; ++index) {
        const int limb = column.toggle_nonzero_limbs[static_cast<std::size_t>(index)];
        const std::size_t limb_index = static_cast<std::size_t>(limb);
        child.det[limb_index] ^= column.active_toggle_detector[limb_index];
    }
}

void fill_toggle_child_key4(Key4& child, const Key4& parent_key, const Column& column, int n_limbs, const LogicalMask& logical) {
    fill_no_child_key4(child, parent_key, column, n_limbs, logical);
    apply_active_toggle_to_child4(child, column);
}

struct Candidate4Better {
    int n_limbs = 0;

    bool operator()(const Candidate4& lhs, const Candidate4& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.logmass != rhs.logmass) return lhs.logmass > rhs.logmass;
        return key4_less(lhs.key, rhs.key, n_limbs);
    }
};

struct Candidate4IntBetter {
    int n_limbs = 0;

    bool operator()(const Candidate4Int& lhs, const Candidate4Int& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.logmass != rhs.logmass) return lhs.logmass > rhs.logmass;
        return key4_less(lhs.key, rhs.key, n_limbs);
    }
};

struct Compact4Workspace {
    std::vector<State4> states;
    std::vector<Candidate4> merged;
    std::vector<std::size_t> survivor_indices;
    std::vector<std::size_t> merge_bucket_indices;
    std::vector<std::uint32_t> merge_bucket_generations;
    std::uint32_t merge_generation = 1;
};

DecodeResult decode_native_compact4_with_workspace(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    Compact4Workspace& workspace
) {
    const double total_started = now_seconds();
    DecodeResult result;
    result.stats.no_path_count = 0;

    std::vector<State4>& states = workspace.states;
    std::vector<Candidate4>& merged = workspace.merged;
    std::vector<std::size_t>& survivor_indices = workspace.survivor_indices;
    std::vector<std::size_t>& merge_bucket_indices = workspace.merge_bucket_indices;
    std::vector<std::uint32_t>& merge_bucket_generations = workspace.merge_bucket_generations;
    std::uint32_t& merge_generation = workspace.merge_generation;
    const bool disable_one_pass_prune = one_pass_prune_disabled();
    const std::size_t one_pass_prune_min = one_pass_prune_min_candidates();
    const bool disable_final_prune_sort = final_prune_sort_disabled();
    const bool disable_close_empty_split_merge = compact_close_empty_split_merge_disabled();
    const bool disable_no_merge_transition = no_merge_transition_disabled();
    const bool disable_single_parent_step = single_parent_step_disabled();
    const bool use_single_parent_step =
        !disable_single_parent_step && !model.force_full_key && model.n_limbs > kCompactLimbs && model.active4_supported;
    const bool profile_enabled = native_profile_enabled();

    states.clear();
    states.push_back(State4{Key4{}, 0.0, 0.0});

    for (std::size_t column_index = 0; column_index < model.columns.size(); ++column_index) {
        const Column& column = model.columns[column_index];
        result.stats.processed_columns = static_cast<int>(column_index) + 1;
        merged.clear();
        merged.reserve(states.size() * 2);
        std::size_t merge_bucket_mask = 0;
        bool use_merge_index = false;

        auto emit_child = [&](const Key4& key, double logmass, double parity_logodds) {
            if (profile_enabled) result.stats.profile_emit_child_calls += 1ULL;
            if (!use_merge_index) {
                for (Candidate4& existing : merged) {
                    if (key4_equal(existing.key, key, model.n_limbs)) {
                        existing.logmass = logaddexp_pair(existing.logmass, logmass);
                        if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                        return;
                    }
                }
                if (merged.size() < kLinearMergeLimit) {
                    merged.push_back(Candidate4{key, logmass, parity_logodds, -std::numeric_limits<double>::infinity()});
                    return;
                }
                const std::size_t bucket_count = merge_bucket_count_for(states.size() * 2 + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < merged.size(); ++index) {
                    std::size_t slot =
                        hash_key4_for_column(
                            merged[index].key,
                            column,
                            model.n_limbs,
                            model.n_logical_limbs
                        ) & merge_bucket_mask;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                use_merge_index = true;
            }
            std::size_t slot =
                hash_key4_for_column(key, column, model.n_limbs, model.n_logical_limbs) & merge_bucket_mask;
            std::uint64_t probes = 1ULL;
            while (true) {
                if (merge_bucket_generations[slot] != merge_generation) {
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    const std::size_t index = merged.size();
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                    merged.push_back(Candidate4{key, logmass, parity_logodds, -std::numeric_limits<double>::infinity()});
                    return;
                }
                const std::size_t existing_index = merge_bucket_indices[slot];
                Candidate4& existing = merged[existing_index];
                if (key4_equal_for_column(existing.key, key, column, model.n_limbs)) {
                    existing.logmass = logaddexp_pair(existing.logmass, logmass);
                    if (profile_enabled) {
                        result.stats.profile_merge_duplicate_count += 1ULL;
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    return;
                }
                slot = (slot + 1U) & merge_bucket_mask;
                probes += 1ULL;
            }
        };

        const double transition_started = model.collect_phase_timing ? now_seconds() : 0.0;
        const bool toggle_finite = std::isfinite(column.toggle_logodds);
        const bool close_empty = column.close_nonzero_count == 0;
        result.stats.transition_evals +=
            static_cast<std::uint64_t>(states.size()) * static_cast<std::uint64_t>(toggle_finite ? 2 : 1);
        if (use_single_parent_step && states.size() == 1) {
            const State4 parent = states[0];
            const auto closure_acceptance =
                closure_acceptance_sparse4(parent.key, column, syndrome, toggle_finite);
            const bool no_accepted = closure_acceptance.first;
            const bool toggle_accepted = closure_acceptance.second;
            Candidate4 first_candidate;
            Candidate4 second_candidate;
            bool have_first_candidate = false;
            bool have_second_candidate = false;
            auto add_direct_candidate = [&](const Key4& key, double logmass, double parity_logodds) {
                if (!have_first_candidate) {
                    first_candidate = Candidate4{
                        key,
                        logmass,
                        parity_logodds,
                        -std::numeric_limits<double>::infinity(),
                    };
                    have_first_candidate = true;
                    return;
                }
                if (key4_equal(first_candidate.key, key, model.n_limbs)) {
                    first_candidate.logmass = logaddexp_pair(first_candidate.logmass, logmass);
                    return;
                }
                second_candidate = Candidate4{
                    key,
                    logmass,
                    parity_logodds,
                    -std::numeric_limits<double>::infinity(),
                };
                have_second_candidate = true;
            };

            if (no_accepted || toggle_accepted) {
                const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                if (no_accepted) {
                    Key4 no_key;
                    fill_no_child_key4(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                    add_direct_candidate(no_key, base_logmass, parity_pair.first);
                }
                if (toggle_accepted) {
                    Key4 toggle_key;
                    fill_toggle_child_key4(
                        toggle_key,
                        parent.key,
                        column,
                        model.n_limbs,
                        logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs)
                    );
                    add_direct_candidate(
                        toggle_key,
                        base_logmass + column.toggle_logodds,
                        parity_pair.second
                    );
                }
            }
            if (model.collect_phase_timing) {
                result.stats.transition_time_s += now_seconds() - transition_started;
            }

            const std::uint64_t candidate_count =
                static_cast<std::uint64_t>(have_first_candidate ? (have_second_candidate ? 2 : 1) : 0);
            result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
            result.stats.sum_pre_prune_state_count += candidate_count;
            if (candidate_count == 0) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }

            const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
            first_candidate.score = first_candidate.logmass + score_alpha * first_candidate.parity_logodds;
            if (profile_enabled) result.stats.profile_score_evals += 1ULL;
            double best_score = first_candidate.score;
            if (have_second_candidate) {
                second_candidate.score = second_candidate.logmass + score_alpha * second_candidate.parity_logodds;
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (second_candidate.score > best_score) best_score = second_candidate.score;
            }
            const double cutoff = best_score - Delta;
            const bool keep_first = first_candidate.score >= cutoff;
            const bool keep_second = have_second_candidate && second_candidate.score >= cutoff;

            states.clear();
            Candidate4Better better{model.n_limbs};
            if (keep_first && keep_second && K == 1) {
                const Candidate4& survivor =
                    better(second_candidate, first_candidate) ? second_candidate : first_candidate;
                states.push_back(State4{survivor.key, survivor.logmass, survivor.parity_logodds});
            } else if (keep_first && keep_second && !disable_final_prune_sort) {
                if (better(second_candidate, first_candidate)) {
                    states.push_back(State4{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                    states.push_back(State4{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                } else {
                    states.push_back(State4{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                    states.push_back(State4{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            } else {
                if (keep_first) {
                    states.push_back(State4{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                }
                if (keep_second) {
                    states.push_back(State4{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            }
            if (model.collect_phase_timing) {
                result.stats.prune_time_s += now_seconds() - prune_started;
            }

            const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
            result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
            result.stats.sum_post_prune_state_count += post_count;
            if (states.empty()) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }
            continue;
        }
        if (
            close_empty && toggle_finite && column.no_child_injective4 && column.toggle_has_new_active_bit
                && !disable_no_merge_transition
        ) {
            if (profile_enabled) result.stats.profile_no_merge_transition_columns += 1ULL;
            merged.reserve(states.size() * 2);
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                merged.push_back(Candidate4{
                    parent.key,
                    base_logmass,
                    parity_pair.first,
                    -std::numeric_limits<double>::infinity(),
                });

                Key4 toggle_key = parent.key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child4(toggle_key, column);
                merged.push_back(Candidate4{
                    toggle_key,
                    base_logmass + column.toggle_logodds,
                    parity_pair.second,
                    -std::numeric_limits<double>::infinity(),
                });
            }
        } else if (
            close_empty && toggle_finite && column.no_child_injective4 && !disable_close_empty_split_merge
        ) {
            if (profile_enabled) result.stats.profile_split_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                merged.push_back(Candidate4{
                    parent.key,
                    base_logmass,
                    parity_pair.first,
                    -std::numeric_limits<double>::infinity(),
                });
            }

            const std::size_t no_count = merged.size();
            if (no_count < kLinearMergeLimit) {
                for (const State4& parent : states) {
                    const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                    Key4 toggle_key = parent.key;
                    logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child4(toggle_key, column);
                    const Candidate4 toggle_candidate{
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second,
                        -std::numeric_limits<double>::infinity(),
                    };
                    bool matched = false;
                    for (std::size_t no_index = 0; no_index < merged.size(); ++no_index) {
                        Candidate4& existing = merged[no_index];
                        if (key4_equal(existing.key, toggle_candidate.key, model.n_limbs)) {
                            existing.logmass = logaddexp_pair(existing.logmass, toggle_candidate.logmass);
                            if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        merged.push_back(toggle_candidate);
                    }
                }
            } else {
                const std::size_t bucket_count = merge_bucket_count_for(no_count + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < no_count; ++index) {
                    std::size_t slot =
                        hash_key4_for_column(
                            merged[index].key,
                            column,
                            model.n_limbs,
                            model.n_logical_limbs
                        ) & merge_bucket_mask;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                for (const State4& parent : states) {
                    const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                    Key4 toggle_key = parent.key;
                    logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child4(toggle_key, column);
                    const Candidate4 toggle_candidate{
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second,
                        -std::numeric_limits<double>::infinity(),
                    };
                    std::size_t slot =
                        hash_key4_for_column(
                            toggle_candidate.key,
                            column,
                            model.n_limbs,
                            model.n_logical_limbs
                        ) & merge_bucket_mask;
                    bool matched = false;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        const std::size_t existing_index = merge_bucket_indices[slot];
                        Candidate4& existing = merged[existing_index];
                        if (key4_equal_for_column(existing.key, toggle_candidate.key, column, model.n_limbs)) {
                            existing.logmass = logaddexp_pair(existing.logmass, toggle_candidate.logmass);
                            if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                            matched = true;
                            break;
                        }
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    if (!matched) {
                        const std::size_t index = merged.size();
                        merge_bucket_generations[slot] = merge_generation;
                        merge_bucket_indices[slot] = index;
                        merged.push_back(toggle_candidate);
                    }
                }
            }
        } else if (close_empty && toggle_finite) {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                Key4 toggle_key = parent.key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child4(toggle_key, column);

                const double base_logmass = parent.logmass + column.no_error_log_const;
                emit_child(parent.key, base_logmass, parity_pair.first);
                emit_child(toggle_key, base_logmass + column.toggle_logodds, parity_pair.second);
            }
        } else if (close_empty) {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_fast4(parent, column, syndrome);
                emit_child(parent.key, parent.logmass + column.no_error_log_const, parity_pair.first);
            }
        } else {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto closure_acceptance =
                    closure_acceptance_sparse4(parent.key, column, syndrome, toggle_finite);
                const bool no_accepted = closure_acceptance.first;
                const bool toggle_accepted = closure_acceptance.second;

                Key4 no_key;
                Key4 toggle_key;
                const LogicalMask toggle_logical =
                    logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
                if (no_accepted && toggle_accepted) {
                    fill_no_child_key4(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                    toggle_key = no_key;
                    toggle_key.logical = toggle_logical;
                    apply_active_toggle_to_child4(toggle_key, column);
                } else if (no_accepted) {
                    fill_no_child_key4(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                } else if (toggle_accepted) {
                    fill_toggle_child_key4(toggle_key, parent.key, column, model.n_limbs, toggle_logical);
                }

                std::pair<double, double> parity_pair;
                if (no_accepted || toggle_accepted) {
                    parity_pair = child_parities_fast4(parent, column, syndrome);
                }

                if (no_accepted) {
                    emit_child(no_key, parent.logmass + column.no_error_log_const, parity_pair.first);
                }
                if (toggle_accepted) {
                    emit_child(
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second
                    );
                }
            }
        }
        if (model.collect_phase_timing) {
            result.stats.transition_time_s += now_seconds() - transition_started;
        }

        const std::uint64_t candidate_count = static_cast<std::uint64_t>(merged.size());
        result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
        result.stats.sum_pre_prune_state_count += candidate_count;
        if (merged.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }

        const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
        double best_score = -std::numeric_limits<double>::infinity();
        survivor_indices.clear();
        const bool use_one_pass_prune =
            !disable_one_pass_prune && merged.size() >= one_pass_prune_min;
        if (!use_one_pass_prune) {
            for (Candidate4& candidate : merged) {
                candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (candidate.score > best_score) best_score = candidate.score;
            }
            const double cutoff = best_score - Delta;
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                if (merged[index].score >= cutoff) {
                    survivor_indices.push_back(index);
                }
            }
        } else {
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                Candidate4& candidate = merged[index];
                candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (candidate.score > best_score) best_score = candidate.score;
                const double running_cutoff = best_score - Delta;
                if (candidate.score >= running_cutoff) {
                    survivor_indices.push_back(index);
                }
            }
            const double cutoff = best_score - Delta;
            std::size_t write_index = 0;
            for (std::size_t read_index = 0; read_index < survivor_indices.size(); ++read_index) {
                const std::size_t candidate_index = survivor_indices[read_index];
                if (merged[candidate_index].score >= cutoff) {
                    survivor_indices[write_index] = candidate_index;
                    write_index += 1U;
                }
            }
            survivor_indices.resize(write_index);
        }
        Candidate4Better better{model.n_limbs};
        auto better_index = [&](std::size_t lhs, std::size_t rhs) {
            return better(merged[lhs], merged[rhs]);
        };
        if (static_cast<int>(survivor_indices.size()) > K) {
            if (profile_enabled) result.stats.profile_nth_element_calls += 1ULL;
            std::nth_element(
                survivor_indices.begin(),
                survivor_indices.begin() + K,
                survivor_indices.end(),
                better_index
            );
            survivor_indices.resize(static_cast<std::size_t>(K));
        }
        if (!disable_final_prune_sort) {
            std::sort(survivor_indices.begin(), survivor_indices.end(), better_index);
        }
        states.clear();
        states.reserve(survivor_indices.size());
        for (std::size_t survivor_index : survivor_indices) {
            const Candidate4& survivor = merged[survivor_index];
            states.push_back(State4{survivor.key, survivor.logmass, survivor.parity_logodds});
        }
        if (model.collect_phase_timing) {
            result.stats.prune_time_s += now_seconds() - prune_started;
        }

        const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
        result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
        result.stats.sum_post_prune_state_count += post_count;
        if (states.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }
    }

    for (const State4& state : states) {
        if (!det4_zero(state.key, model.n_limbs)) continue;
        auto found = result.terminal_log_masses.find(state.key.logical);
        if (found == result.terminal_log_masses.end()) {
            result.terminal_log_masses[state.key.logical] = state.logmass;
        } else {
            found->second = logaddexp_pair(found->second, state.logmass);
        }
    }

    if (result.terminal_log_masses.empty()) {
        result.ok = false;
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        return result;
    }

    result.ok = true;
    double top1 = -std::numeric_limits<double>::infinity();
    double top2 = -std::numeric_limits<double>::infinity();
    bool have_hat = false;
    result.log_evidence = -std::numeric_limits<double>::infinity();
    for (const auto& item : result.terminal_log_masses) {
        const LogicalMask& logical = item.first;
        const double log_mass = item.second;
        result.log_evidence = logaddexp_pair(result.log_evidence, log_mass);
        if (!have_hat || log_mass > top1 || (log_mass == top1 && logical < result.logical_hat)) {
            top2 = top1;
            top1 = log_mass;
            result.logical_hat = logical;
            have_hat = true;
        } else if (log_mass > top2) {
            top2 = log_mass;
        }
    }
    result.terminal_top_log_mass_gap =
        std::isfinite(top2) ? (top1 - top2) : std::numeric_limits<double>::infinity();
    result.stats.no_path_count = 0;
    result.stats.total_time_s = now_seconds() - total_started;
    return result;
}

DecodeResult decode_native_compact4(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha
) {
    Compact4Workspace workspace;
    return decode_native_compact4_with_workspace(model, syndrome, K, Delta, score_alpha, workspace);
}

std::uint64_t active4_parent_term_limb(const State4& parent, const RowTerm& term) {
    if (term.active4_slot < 0) return 0;
    return parent.key.det[static_cast<std::size_t>(term.active4_slot)];
}

std::uint64_t active4_parent_term_limb_int(const State4Int& parent, const RowTerm& term) {
    if (term.active4_slot < 0) return 0;
    return parent.key.det[static_cast<std::size_t>(term.active4_slot)];
}

std::size_t local_pattern_bit_active4(
    const State4& parent,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const RowTerm& term,
    std::size_t index
) {
    const std::uint64_t bit =
        (active4_parent_term_limb(parent, term) ^ syndrome[static_cast<std::size_t>(term.limb)])
        & term.bit;
    return bit != 0 ? (static_cast<std::size_t>(1) << index) : 0U;
}

std::size_t local_pattern_index_active4(
    const State4& parent,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const LocalPatternTable& table
) {
    if (table.small_enabled) {
        std::size_t pattern = 0;
        switch (table.row_count) {
            case 6:
                pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[5], 5);
                [[fallthrough]];
            case 5:
                pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[4], 4);
                [[fallthrough]];
            case 4:
                pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[3], 3);
                [[fallthrough]];
            case 3:
                pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[2], 2);
                [[fallthrough]];
            case 2:
                pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[1], 1);
                [[fallthrough]];
            case 1:
                pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[0], 0);
                break;
            default:
                break;
        }
        return pattern;
    }
    std::size_t pattern = 0;
    for (std::size_t index = 0; index < static_cast<std::size_t>(table.row_count); ++index) {
        pattern |= local_pattern_bit_active4(parent, syndrome, table.rows[index], index);
    }
    return pattern;
}

std::size_t local_pattern_bit_active4_int(
    const State4Int& parent,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const RowTerm& term,
    std::size_t index
) {
    const std::uint64_t bit =
        (active4_parent_term_limb_int(parent, term) ^ syndrome[static_cast<std::size_t>(term.limb)])
        & term.bit;
    return bit != 0 ? (static_cast<std::size_t>(1) << index) : 0U;
}

std::size_t local_pattern_index_active4_int(
    const State4Int& parent,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    const LocalPatternTable& table
) {
    if (table.small_enabled) {
        std::size_t pattern = 0;
        switch (table.row_count) {
            case 6:
                pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[5], 5);
                [[fallthrough]];
            case 5:
                pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[4], 4);
                [[fallthrough]];
            case 4:
                pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[3], 3);
                [[fallthrough]];
            case 3:
                pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[2], 2);
                [[fallthrough]];
            case 2:
                pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[1], 1);
                [[fallthrough]];
            case 1:
                pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[0], 0);
                break;
            default:
                break;
        }
        return pattern;
    }
    std::size_t pattern = 0;
    for (std::size_t index = 0; index < static_cast<std::size_t>(table.row_count); ++index) {
        pattern |= local_pattern_bit_active4_int(parent, syndrome, table.rows[index], index);
    }
    return pattern;
}

bool active4_key_less_after(const Key4& lhs, const Key4& rhs, const Column& column, int n_logical_limbs) {
    if (!logical_equal_limited(lhs.logical, rhs.logical, n_logical_limbs)) {
        return logical_less_limited(lhs.logical, rhs.logical, n_logical_limbs);
    }
    for (int slot = column.active_nonzero_count - 1; slot >= 0; --slot) {
        const std::size_t slot_index = static_cast<std::size_t>(slot);
        if (lhs.det[slot_index] != rhs.det[slot_index]) {
            return lhs.det[slot_index] < rhs.det[slot_index];
        }
    }
    return false;
}

bool active4_key_equal_after(const Key4& lhs, const Key4& rhs, const Column& column, int n_logical_limbs) {
    if (!logical_equal_limited(lhs.logical, rhs.logical, n_logical_limbs)) return false;
    for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
        const std::size_t slot_index = static_cast<std::size_t>(slot);
        if (lhs.det[slot_index] != rhs.det[slot_index]) return false;
    }
    return true;
}

std::size_t active4_hash_after(const Key4& key, const Column& column, int n_logical_limbs) {
    std::uint64_t h = static_cast<std::uint64_t>(hash_logical(key.logical, n_logical_limbs));
    for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
        const std::uint64_t value = mix_hash_word(key.det[static_cast<std::size_t>(slot)]);
        h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<std::size_t>(h);
}

std::pair<bool, bool> closure_acceptance_active4(
    const State4& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    bool toggle_finite
) {
    bool no_accepted = true;
    bool toggle_accepted = toggle_finite;
    for (int index = 0; index < column.close_nonzero_count; ++index) {
        const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
        const std::size_t limb_index = static_cast<std::size_t>(limb);
        const std::uint64_t close_mask = column.close_mask[limb_index];
        const std::uint64_t parent_mismatch =
            ((column.before_slot_by_limb4[limb_index] < 0
                  ? 0
                  : parent.key.det[static_cast<std::size_t>(column.before_slot_by_limb4[limb_index])])
             ^ syndrome[limb_index]) & close_mask;
        if (parent_mismatch != 0) {
            no_accepted = false;
        }
        if (toggle_finite) {
            const std::uint64_t toggle_mismatch =
                parent_mismatch ^ (column.toggle_detector[limb_index] & close_mask);
            if (toggle_mismatch != 0) {
                toggle_accepted = false;
            }
        }
        if (!no_accepted && (!toggle_finite || !toggle_accepted)) {
            break;
        }
    }
    return {no_accepted, toggle_accepted};
}

std::pair<bool, bool> closure_acceptance_active4_int(
    const State4Int& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    bool toggle_finite
) {
    bool no_accepted = true;
    bool toggle_accepted = toggle_finite;
    for (int index = 0; index < column.close_nonzero_count; ++index) {
        const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
        const std::size_t limb_index = static_cast<std::size_t>(limb);
        const std::uint64_t close_mask = column.close_mask[limb_index];
        const std::uint64_t parent_mismatch =
            ((column.before_slot_by_limb4[limb_index] < 0
                  ? 0
                  : parent.key.det[static_cast<std::size_t>(column.before_slot_by_limb4[limb_index])])
             ^ syndrome[limb_index]) & close_mask;
        if (parent_mismatch != 0) {
            no_accepted = false;
        }
        if (toggle_finite) {
            const std::uint64_t toggle_mismatch =
                parent_mismatch ^ (column.toggle_detector[limb_index] & close_mask);
            if (toggle_mismatch != 0) {
                toggle_accepted = false;
            }
        }
        if (!no_accepted && (!toggle_finite || !toggle_accepted)) {
            break;
        }
    }
    return {no_accepted, toggle_accepted};
}

std::pair<double, double> child_parities_active4(
    const State4& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    if (column.pattern_table.enabled) {
        const std::size_t pattern = local_pattern_index_active4(parent, syndrome, column.pattern_table);
        return {
            parent.parity_logodds + local_pattern_no_delta(column.pattern_table, pattern),
            parent.parity_logodds + local_pattern_toggle_delta(column.pattern_table, pattern),
        };
    }

    double base = parent.parity_logodds;
    for (const RowTerm& term : column.before_terms) {
        const std::uint64_t bit =
            (active4_parent_term_limb(parent, term) ^ syndrome[static_cast<std::size_t>(term.limb)])
            & term.bit;
        if (bit != 0) base -= term.parity;
    }
    double no_parity = base;
    double toggle_parity = base;
    for (const RowTerm& term : column.after_terms) {
        const std::uint64_t bit =
            (active4_parent_term_limb(parent, term) ^ syndrome[static_cast<std::size_t>(term.limb)])
            & term.bit;
        if (bit != 0) {
            no_parity += term.parity;
        } else {
            toggle_parity += term.parity;
        }
    }
    return {no_parity, toggle_parity};
}

std::pair<std::int64_t, std::int64_t> child_parities_active4_int(
    const State4Int& parent,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int int_metric_scale
) {
    if (column.pattern_table.enabled) {
        const std::size_t pattern = local_pattern_index_active4_int(parent, syndrome, column.pattern_table);
        return {
            parent.parity_logodds + local_pattern_no_delta_int(column.pattern_table, pattern, int_metric_scale),
            parent.parity_logodds + local_pattern_toggle_delta_int(column.pattern_table, pattern, int_metric_scale),
        };
    }

    std::int64_t base = parent.parity_logodds;
    for (const RowTerm& term : column.before_terms) {
        const std::uint64_t bit =
            (active4_parent_term_limb_int(parent, term) ^ syndrome[static_cast<std::size_t>(term.limb)])
            & term.bit;
        if (bit != 0) {
            base -= row_term_parity_int(term, int_metric_scale);
        }
    }
    std::int64_t no_parity = base;
    std::int64_t toggle_parity = base;
    for (const RowTerm& term : column.after_terms) {
        const std::int64_t parity = row_term_parity_int(term, int_metric_scale);
        const std::uint64_t bit =
            (active4_parent_term_limb_int(parent, term) ^ syndrome[static_cast<std::size_t>(term.limb)])
            & term.bit;
        if (bit != 0) {
            no_parity += parity;
        } else {
            toggle_parity += parity;
        }
    }
    return {no_parity, toggle_parity};
}

void fill_no_child_key_active4(Key4& child, const State4& parent, const Column& column, const LogicalMask& logical) {
    if (column.active4_no_child_identity) {
        child = parent.key;
        child.logical = logical;
        return;
    }
    child.logical = logical;
    for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
        const int before_slot = column.active_before_slots4[static_cast<std::size_t>(slot)];
        std::uint64_t value = 0;
        if (before_slot >= 0) {
            value = parent.key.det[static_cast<std::size_t>(before_slot)]
                & column.active_slot_masks4[static_cast<std::size_t>(slot)];
        }
        child.det[static_cast<std::size_t>(slot)] = value;
    }
}

void fill_no_child_key_active4_int(
    Key4& child,
    const State4Int& parent,
    const Column& column,
    const LogicalMask& logical
) {
    if (column.active4_no_child_identity) {
        child = parent.key;
        child.logical = logical;
        return;
    }
    child.logical = logical;
    for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
        const int before_slot = column.active_before_slots4[static_cast<std::size_t>(slot)];
        std::uint64_t value = 0;
        if (before_slot >= 0) {
            value = parent.key.det[static_cast<std::size_t>(before_slot)]
                & column.active_slot_masks4[static_cast<std::size_t>(slot)];
        }
        child.det[static_cast<std::size_t>(slot)] = value;
    }
}

void apply_active_toggle_to_child_active4(Key4& child, const Column& column) {
    for (int slot = 0; slot < column.active_nonzero_count; ++slot) {
        child.det[static_cast<std::size_t>(slot)] ^= column.active_toggle_slots4[static_cast<std::size_t>(slot)];
    }
}

struct Active4CandidateBetter {
    const Column* column = nullptr;
    int n_logical_limbs = kMaxLogicalLimbs;

    bool operator()(const Candidate4& lhs, const Candidate4& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.logmass != rhs.logmass) return lhs.logmass > rhs.logmass;
        return active4_key_less_after(lhs.key, rhs.key, *column, n_logical_limbs);
    }
};

struct Active4CandidateIntBetter {
    const Column* column = nullptr;
    int n_logical_limbs = kMaxLogicalLimbs;

    bool operator()(const Candidate4Int& lhs, const Candidate4Int& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.logmass != rhs.logmass) return lhs.logmass > rhs.logmass;
        return active4_key_less_after(lhs.key, rhs.key, *column, n_logical_limbs);
    }
};

bool try_active4_small_state_step(
    const NativeModel& model,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    bool disable_final_prune_sort,
    bool profile_enabled,
    double transition_started,
    double total_started,
    std::vector<State4>& states,
    DecodeResult& result,
    bool& no_path
) {
    no_path = false;
    const std::size_t parent_count = states.size();
    if (parent_count == 0 || parent_count > kActive4SmallStateStepLimit) {
        return false;
    }

    std::array<Candidate4, kActive4SmallStateStepLimit * 2U> candidates{};
    std::size_t candidate_count = 0;
    const bool toggle_finite = std::isfinite(column.toggle_logodds);
    auto add_candidate = [&](const Key4& key, double logmass, double parity_logodds) {
        for (std::size_t index = 0; index < candidate_count; ++index) {
            if (active4_key_equal_after(candidates[index].key, key, column, model.n_logical_limbs)) {
                candidates[index].logmass = logaddexp_pair(candidates[index].logmass, logmass);
                return;
            }
        }
        candidates[candidate_count] = Candidate4{
            key,
            logmass,
            parity_logodds,
            -std::numeric_limits<double>::infinity(),
        };
        candidate_count += 1U;
    };

    for (const State4& parent : states) {
        const auto closure_acceptance =
            closure_acceptance_active4(parent, column, syndrome, toggle_finite);
        const bool no_accepted = closure_acceptance.first;
        const bool toggle_accepted = closure_acceptance.second;
        if (!no_accepted && !toggle_accepted) {
            continue;
        }
        const auto parity_pair = child_parities_active4(parent, column, syndrome);
        const double base_logmass = parent.logmass + column.no_error_log_const;
        Key4 no_key;
        fill_no_child_key_active4(no_key, parent, column, parent.key.logical);
        if (no_accepted) {
            add_candidate(no_key, base_logmass, parity_pair.first);
        }
        if (toggle_accepted) {
            Key4 toggle_key = no_key;
            toggle_key.logical = logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
            apply_active_toggle_to_child_active4(toggle_key, column);
            add_candidate(toggle_key, base_logmass + column.toggle_logodds, parity_pair.second);
        }
    }

    if (model.collect_phase_timing) {
        result.stats.transition_time_s += now_seconds() - transition_started;
    }
    result.stats.max_pre_prune_state_count = std::max(
        result.stats.max_pre_prune_state_count,
        static_cast<std::uint64_t>(candidate_count)
    );
    result.stats.sum_pre_prune_state_count += static_cast<std::uint64_t>(candidate_count);
    if (candidate_count == 0) {
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        result.ok = false;
        no_path = true;
        return true;
    }

    const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < candidate_count; ++index) {
        candidates[index].score = candidates[index].logmass + score_alpha * candidates[index].parity_logodds;
        if (profile_enabled) result.stats.profile_score_evals += 1ULL;
        if (candidates[index].score > best_score) {
            best_score = candidates[index].score;
        }
    }

    const double cutoff = best_score - Delta;
    std::array<std::size_t, kActive4SmallStateStepLimit * 2U> survivor_indices{};
    std::size_t survivor_count = 0;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        if (candidates[index].score >= cutoff) {
            survivor_indices[survivor_count] = index;
            survivor_count += 1U;
        }
    }

    Active4CandidateBetter better{&column, model.n_logical_limbs};
    auto better_index = [&](std::size_t lhs, std::size_t rhs) {
        return better(candidates[lhs], candidates[rhs]);
    };
    if (survivor_count > static_cast<std::size_t>(K)) {
        std::sort(
            survivor_indices.begin(),
            survivor_indices.begin() + static_cast<std::ptrdiff_t>(survivor_count),
            better_index
        );
        survivor_count = static_cast<std::size_t>(K);
    } else if (!disable_final_prune_sort) {
        std::sort(
            survivor_indices.begin(),
            survivor_indices.begin() + static_cast<std::ptrdiff_t>(survivor_count),
            better_index
        );
    }

    states.clear();
    states.reserve(survivor_count);
    for (std::size_t index = 0; index < survivor_count; ++index) {
        const Candidate4& survivor = candidates[survivor_indices[index]];
        states.push_back(State4{survivor.key, survivor.logmass, survivor.parity_logodds});
    }
    if (model.collect_phase_timing) {
        result.stats.prune_time_s += now_seconds() - prune_started;
    }

    const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
    result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
    result.stats.sum_post_prune_state_count += post_count;
    if (states.empty()) {
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        result.ok = false;
        no_path = true;
    }
    return true;
}

DecodeResult decode_native_active4_with_workspace(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    Compact4Workspace& workspace
) {
    const double total_started = now_seconds();
    DecodeResult result;
    result.stats.no_path_count = 0;

    std::vector<State4>& states = workspace.states;
    std::vector<Candidate4>& merged = workspace.merged;
    std::vector<std::size_t>& survivor_indices = workspace.survivor_indices;
    std::vector<std::size_t>& merge_bucket_indices = workspace.merge_bucket_indices;
    std::vector<std::uint32_t>& merge_bucket_generations = workspace.merge_bucket_generations;
    std::uint32_t& merge_generation = workspace.merge_generation;
    const bool disable_one_pass_prune = one_pass_prune_disabled();
    const std::size_t one_pass_prune_min = one_pass_prune_min_candidates();
    const bool disable_final_prune_sort = final_prune_sort_disabled();
    const bool disable_close_empty_split_merge = close_empty_split_merge_disabled();
    const bool disable_no_merge_transition = no_merge_transition_disabled();
    const bool disable_single_parent_step = single_parent_step_disabled();
    const bool disable_small_state_step = small_state_step_disabled();
    const bool use_single_parent_step =
        !disable_single_parent_step && !model.force_full_key && model.n_limbs > kCompactLimbs && model.active4_supported;
    const bool use_small_state_step =
        !disable_small_state_step && use_single_parent_step;
    const bool profile_enabled = native_profile_enabled();

    states.clear();
    states.push_back(State4{Key4{}, 0.0, 0.0});

    for (std::size_t column_index = 0; column_index < model.columns.size(); ++column_index) {
        const Column& column = model.columns[column_index];
        result.stats.processed_columns = static_cast<int>(column_index) + 1;
        merged.clear();
        merged.reserve(states.size() * 2);
        std::size_t merge_bucket_mask = 0;
        bool use_merge_index = false;

        auto emit_child = [&](const Key4& key, double logmass, double parity_logodds) {
            if (profile_enabled) result.stats.profile_emit_child_calls += 1ULL;
            if (!use_merge_index) {
                for (Candidate4& existing : merged) {
                    if (active4_key_equal_after(existing.key, key, column, model.n_logical_limbs)) {
                        existing.logmass = logaddexp_pair(existing.logmass, logmass);
                        if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                        return;
                    }
                }
                if (merged.size() < kLinearMergeLimit) {
                    merged.push_back(Candidate4{key, logmass, parity_logodds, -std::numeric_limits<double>::infinity()});
                    return;
                }
                const std::size_t bucket_count = merge_bucket_count_for(states.size() * 2 + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < merged.size(); ++index) {
                    std::size_t slot =
                        active4_hash_after(merged[index].key, column, model.n_logical_limbs) & merge_bucket_mask;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                use_merge_index = true;
            }
            std::size_t slot = active4_hash_after(key, column, model.n_logical_limbs) & merge_bucket_mask;
            std::uint64_t probes = 1ULL;
            while (true) {
                if (merge_bucket_generations[slot] != merge_generation) {
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    const std::size_t index = merged.size();
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                    merged.push_back(Candidate4{key, logmass, parity_logodds, -std::numeric_limits<double>::infinity()});
                    return;
                }
                const std::size_t existing_index = merge_bucket_indices[slot];
                Candidate4& existing = merged[existing_index];
                if (active4_key_equal_after(existing.key, key, column, model.n_logical_limbs)) {
                    existing.logmass = logaddexp_pair(existing.logmass, logmass);
                    if (profile_enabled) {
                        result.stats.profile_merge_duplicate_count += 1ULL;
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    return;
                }
                slot = (slot + 1U) & merge_bucket_mask;
                probes += 1ULL;
            }
        };

        const double transition_started = model.collect_phase_timing ? now_seconds() : 0.0;
        const bool toggle_finite = std::isfinite(column.toggle_logodds);
        const bool close_empty = column.close_nonzero_count == 0;
        result.stats.transition_evals +=
            static_cast<std::uint64_t>(states.size()) * static_cast<std::uint64_t>(toggle_finite ? 2 : 1);
        if (
            use_small_state_step
            && states.size() >= 2
            && states.size() <= kActive4SmallStateStepLimit
        ) {
            bool no_path = false;
            if (try_active4_small_state_step(
                    model,
                    column,
                    syndrome,
                    K,
                    Delta,
                    score_alpha,
                    disable_final_prune_sort,
                    profile_enabled,
                    transition_started,
                    total_started,
                    states,
                    result,
                    no_path
                )) {
                if (no_path) {
                    return result;
                }
                continue;
            }
        }
        if (use_single_parent_step && states.size() == 1) {
            const State4 parent = states[0];
            const auto closure_acceptance =
                closure_acceptance_active4(parent, column, syndrome, toggle_finite);
            const bool no_accepted = closure_acceptance.first;
            const bool toggle_accepted = closure_acceptance.second;
            Candidate4 first_candidate;
            Candidate4 second_candidate;
            bool have_first_candidate = false;
            bool have_second_candidate = false;
            auto add_direct_candidate = [&](const Key4& key, double logmass, double parity_logodds) {
                if (!have_first_candidate) {
                    first_candidate = Candidate4{
                        key,
                        logmass,
                        parity_logodds,
                        -std::numeric_limits<double>::infinity(),
                    };
                    have_first_candidate = true;
                    return;
                }
                if (active4_key_equal_after(first_candidate.key, key, column, model.n_logical_limbs)) {
                    first_candidate.logmass = logaddexp_pair(first_candidate.logmass, logmass);
                    return;
                }
                second_candidate = Candidate4{
                    key,
                    logmass,
                    parity_logodds,
                    -std::numeric_limits<double>::infinity(),
                };
                have_second_candidate = true;
            };

            if (no_accepted || toggle_accepted) {
                const auto parity_pair = child_parities_active4(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                Key4 no_key;
                if (no_accepted || toggle_accepted) {
                    fill_no_child_key_active4(no_key, parent, column, parent.key.logical);
                }
                if (no_accepted) {
                    add_direct_candidate(no_key, base_logmass, parity_pair.first);
                }
                if (toggle_accepted) {
                    Key4 toggle_key = no_key;
                    toggle_key.logical = logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                    add_direct_candidate(
                        toggle_key,
                        base_logmass + column.toggle_logodds,
                        parity_pair.second
                    );
                }
            }
            if (model.collect_phase_timing) {
                result.stats.transition_time_s += now_seconds() - transition_started;
            }

            const std::uint64_t candidate_count =
                static_cast<std::uint64_t>(have_first_candidate ? (have_second_candidate ? 2 : 1) : 0);
            result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
            result.stats.sum_pre_prune_state_count += candidate_count;
            if (candidate_count == 0) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }

            const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
            first_candidate.score = first_candidate.logmass + score_alpha * first_candidate.parity_logodds;
            if (profile_enabled) result.stats.profile_score_evals += 1ULL;
            double best_score = first_candidate.score;
            if (have_second_candidate) {
                second_candidate.score = second_candidate.logmass + score_alpha * second_candidate.parity_logodds;
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (second_candidate.score > best_score) best_score = second_candidate.score;
            }
            const double cutoff = best_score - Delta;
            const bool keep_first = first_candidate.score >= cutoff;
            const bool keep_second = have_second_candidate && second_candidate.score >= cutoff;

            states.clear();
            Active4CandidateBetter better{&column, model.n_logical_limbs};
            if (keep_first && keep_second && K == 1) {
                const Candidate4& survivor =
                    better(second_candidate, first_candidate) ? second_candidate : first_candidate;
                states.push_back(State4{survivor.key, survivor.logmass, survivor.parity_logodds});
            } else if (keep_first && keep_second && !disable_final_prune_sort) {
                if (better(second_candidate, first_candidate)) {
                    states.push_back(State4{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                    states.push_back(State4{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                } else {
                    states.push_back(State4{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                    states.push_back(State4{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            } else {
                if (keep_first) {
                    states.push_back(State4{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                }
                if (keep_second) {
                    states.push_back(State4{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            }
            if (model.collect_phase_timing) {
                result.stats.prune_time_s += now_seconds() - prune_started;
            }

            const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
            result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
            result.stats.sum_post_prune_state_count += post_count;
            if (states.empty()) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }
            continue;
        }
        if (
            close_empty && toggle_finite && column.no_child_injective4 && column.toggle_has_new_active_bit
                && !disable_no_merge_transition
        ) {
            if (profile_enabled) result.stats.profile_no_merge_transition_columns += 1ULL;
            merged.reserve(states.size() * 2);
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_active4(parent, column, syndrome);
                Key4 no_key;
                fill_no_child_key_active4(no_key, parent, column, parent.key.logical);

                const double base_logmass = parent.logmass + column.no_error_log_const;
                merged.push_back(Candidate4{
                    no_key,
                    base_logmass,
                    parity_pair.first,
                    -std::numeric_limits<double>::infinity(),
                });

                Key4 toggle_key = no_key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child_active4(toggle_key, column);
                merged.push_back(Candidate4{
                    toggle_key,
                    base_logmass + column.toggle_logodds,
                    parity_pair.second,
                    -std::numeric_limits<double>::infinity(),
                });
            }
        } else if (
            close_empty && toggle_finite && column.no_child_injective4 && !disable_close_empty_split_merge
        ) {
            if (profile_enabled) result.stats.profile_split_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_active4(parent, column, syndrome);
                Key4 no_key;
                fill_no_child_key_active4(no_key, parent, column, parent.key.logical);

                const double base_logmass = parent.logmass + column.no_error_log_const;
                merged.push_back(Candidate4{
                    no_key,
                    base_logmass,
                    parity_pair.first,
                    -std::numeric_limits<double>::infinity(),
                });
            }

            const std::size_t no_count = merged.size();
            if (no_count < kLinearMergeLimit) {
                for (const State4& parent : states) {
                    const auto parity_pair = child_parities_active4(parent, column, syndrome);
                    Key4 toggle_key;
                    fill_no_child_key_active4(toggle_key, parent, column, parent.key.logical);
                    logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                    const Candidate4 toggle_candidate{
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second,
                        -std::numeric_limits<double>::infinity(),
                    };
                    bool matched = false;
                    for (std::size_t no_index = 0; no_index < merged.size(); ++no_index) {
                        Candidate4& existing = merged[no_index];
                        if (active4_key_equal_after(
                                existing.key,
                                toggle_candidate.key,
                                column,
                                model.n_logical_limbs
                            )) {
                            existing.logmass = logaddexp_pair(existing.logmass, toggle_candidate.logmass);
                            if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        merged.push_back(toggle_candidate);
                    }
                }
            } else {
                const std::size_t bucket_count = merge_bucket_count_for(no_count + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < no_count; ++index) {
                    std::size_t slot =
                        active4_hash_after(merged[index].key, column, model.n_logical_limbs) & merge_bucket_mask;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                for (const State4& parent : states) {
                    const auto parity_pair = child_parities_active4(parent, column, syndrome);
                    Key4 toggle_key;
                    fill_no_child_key_active4(toggle_key, parent, column, parent.key.logical);
                    logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                    const Candidate4 toggle_candidate{
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second,
                        -std::numeric_limits<double>::infinity(),
                    };
                    std::size_t slot =
                        active4_hash_after(toggle_candidate.key, column, model.n_logical_limbs) & merge_bucket_mask;
                    bool matched = false;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        const std::size_t existing_index = merge_bucket_indices[slot];
                        Candidate4& existing = merged[existing_index];
                        if (active4_key_equal_after(
                                existing.key,
                                toggle_candidate.key,
                                column,
                                model.n_logical_limbs
                            )) {
                            existing.logmass = logaddexp_pair(existing.logmass, toggle_candidate.logmass);
                            if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                            matched = true;
                            break;
                        }
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    if (!matched) {
                        const std::size_t index = merged.size();
                        merge_bucket_generations[slot] = merge_generation;
                        merge_bucket_indices[slot] = index;
                        merged.push_back(toggle_candidate);
                    }
                }
            }
        } else if (close_empty && toggle_finite) {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_active4(parent, column, syndrome);
                Key4 no_key;
                fill_no_child_key_active4(no_key, parent, column, parent.key.logical);
                Key4 toggle_key = no_key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child_active4(toggle_key, column);

                const double base_logmass = parent.logmass + column.no_error_log_const;
                emit_child(no_key, base_logmass, parity_pair.first);
                emit_child(toggle_key, base_logmass + column.toggle_logodds, parity_pair.second);
            }
        } else if (close_empty) {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto parity_pair = child_parities_active4(parent, column, syndrome);
                Key4 no_key;
                fill_no_child_key_active4(no_key, parent, column, parent.key.logical);
                emit_child(no_key, parent.logmass + column.no_error_log_const, parity_pair.first);
            }
        } else {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4& parent : states) {
                const auto closure_acceptance =
                    closure_acceptance_active4(parent, column, syndrome, toggle_finite);
                const bool no_accepted = closure_acceptance.first;
                const bool toggle_accepted = closure_acceptance.second;

                Key4 no_key;
                Key4 toggle_key;
                if (no_accepted || toggle_accepted) {
                    fill_no_child_key_active4(no_key, parent, column, parent.key.logical);
                }
                if (toggle_accepted) {
                    toggle_key = no_key;
                    toggle_key.logical = logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                }

                std::pair<double, double> parity_pair;
                if (no_accepted || toggle_accepted) {
                    parity_pair = child_parities_active4(parent, column, syndrome);
                }
                if (no_accepted) {
                    emit_child(no_key, parent.logmass + column.no_error_log_const, parity_pair.first);
                }
                if (toggle_accepted) {
                    emit_child(
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second
                    );
                }
            }
        }
        if (model.collect_phase_timing) {
            result.stats.transition_time_s += now_seconds() - transition_started;
        }

        const std::uint64_t candidate_count = static_cast<std::uint64_t>(merged.size());
        result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
        result.stats.sum_pre_prune_state_count += candidate_count;
        if (merged.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }

        const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
        double best_score = -std::numeric_limits<double>::infinity();
        survivor_indices.clear();
        const bool use_one_pass_prune =
            !disable_one_pass_prune && merged.size() >= one_pass_prune_min;
        if (!use_one_pass_prune) {
            for (Candidate4& candidate : merged) {
                candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (candidate.score > best_score) best_score = candidate.score;
            }
            const double cutoff = best_score - Delta;
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                if (merged[index].score >= cutoff) survivor_indices.push_back(index);
            }
        } else {
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                Candidate4& candidate = merged[index];
                candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (candidate.score > best_score) best_score = candidate.score;
                if (candidate.score >= best_score - Delta) {
                    survivor_indices.push_back(index);
                }
            }
            const double cutoff = best_score - Delta;
            std::size_t write_index = 0;
            for (std::size_t read_index = 0; read_index < survivor_indices.size(); ++read_index) {
                const std::size_t candidate_index = survivor_indices[read_index];
                if (merged[candidate_index].score >= cutoff) {
                    survivor_indices[write_index] = candidate_index;
                    write_index += 1U;
                }
            }
            survivor_indices.resize(write_index);
        }
        Active4CandidateBetter better{&column, model.n_logical_limbs};
        auto better_index = [&](std::size_t lhs, std::size_t rhs) {
            return better(merged[lhs], merged[rhs]);
        };
        if (static_cast<int>(survivor_indices.size()) > K) {
            if (profile_enabled) result.stats.profile_nth_element_calls += 1ULL;
            std::nth_element(
                survivor_indices.begin(),
                survivor_indices.begin() + K,
                survivor_indices.end(),
                better_index
            );
            survivor_indices.resize(static_cast<std::size_t>(K));
        }
        if (!disable_final_prune_sort) {
            std::sort(survivor_indices.begin(), survivor_indices.end(), better_index);
        }
        states.clear();
        states.reserve(survivor_indices.size());
        for (std::size_t survivor_index : survivor_indices) {
            const Candidate4& survivor = merged[survivor_index];
            states.push_back(State4{survivor.key, survivor.logmass, survivor.parity_logodds});
        }
        if (model.collect_phase_timing) {
            result.stats.prune_time_s += now_seconds() - prune_started;
        }

        const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
        result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
        result.stats.sum_post_prune_state_count += post_count;
        if (states.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }
    }

    const Column& terminal_column = model.columns.empty() ? Column{} : model.columns.back();
    for (const State4& state : states) {
        bool nonzero = false;
        for (int slot = 0; slot < terminal_column.active_nonzero_count; ++slot) {
            if (state.key.det[static_cast<std::size_t>(slot)] != 0) {
                nonzero = true;
                break;
            }
        }
        if (nonzero) continue;
        auto found = result.terminal_log_masses.find(state.key.logical);
        if (found == result.terminal_log_masses.end()) {
            result.terminal_log_masses[state.key.logical] = state.logmass;
        } else {
            found->second = logaddexp_pair(found->second, state.logmass);
        }
    }

    if (result.terminal_log_masses.empty()) {
        result.ok = false;
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        return result;
    }

    result.ok = true;
    double top1 = -std::numeric_limits<double>::infinity();
    double top2 = -std::numeric_limits<double>::infinity();
    bool have_hat = false;
    result.log_evidence = -std::numeric_limits<double>::infinity();
    for (const auto& item : result.terminal_log_masses) {
        const LogicalMask& logical = item.first;
        const double log_mass = item.second;
        result.log_evidence = logaddexp_pair(result.log_evidence, log_mass);
        if (!have_hat || log_mass > top1 || (log_mass == top1 && logical < result.logical_hat)) {
            top2 = top1;
            top1 = log_mass;
            result.logical_hat = logical;
            have_hat = true;
        } else if (log_mass > top2) {
            top2 = log_mass;
        }
    }
    result.terminal_top_log_mass_gap =
        std::isfinite(top2) ? (top1 - top2) : std::numeric_limits<double>::infinity();
    result.stats.no_path_count = 0;
    result.stats.total_time_s = now_seconds() - total_started;
    return result;
}

struct FullWorkspace {
    std::vector<State> states;
    std::vector<Candidate> merged;
    std::vector<Candidate> toggle_candidates;
    std::vector<std::size_t> survivor_indices;
    std::vector<std::size_t> merge_bucket_indices;
    std::vector<std::uint32_t> merge_bucket_generations;
    std::uint32_t merge_generation = 1;
};

struct MaxLogIntWorkspace {
    std::vector<StateInt> states;
    std::vector<CandidateInt> merged;
    std::vector<State4Int> states4;
    std::vector<Candidate4Int> merged4;
    std::vector<std::size_t> survivor_indices;
    std::vector<std::size_t> merge_bucket_indices;
    std::vector<std::uint32_t> merge_bucket_generations;
    std::uint32_t merge_generation = 1;
};

bool try_active4_small_state_step_int(
    const NativeModel& model,
    const Column& column,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    std::int64_t delta_int,
    std::int64_t alpha_int,
    int int_metric_scale,
    bool disable_final_prune_sort,
    bool profile_enabled,
    double transition_started,
    double total_started,
    std::vector<State4Int>& states,
    DecodeResult& result,
    bool& no_path
) {
    no_path = false;
    const std::size_t parent_count = states.size();
    if (parent_count == 0 || parent_count > kActive4SmallStateStepLimit) {
        return false;
    }

    std::array<Candidate4Int, kActive4SmallStateStepLimit * 2U> candidates{};
    std::size_t candidate_count = 0;
    const bool toggle_finite = std::isfinite(column.toggle_logodds);
    const std::int64_t no_error_log_const = column_no_error_log_const_int(column, int_metric_scale);
    const std::int64_t toggle_logodds = column_toggle_logodds_int(column, int_metric_scale);
    auto add_candidate = [&](const Key4& key, std::int64_t logmass, std::int64_t parity_logodds) {
        for (std::size_t index = 0; index < candidate_count; ++index) {
            if (active4_key_equal_after(candidates[index].key, key, column, model.n_logical_limbs)) {
                if (logmass > candidates[index].logmass) {
                    candidates[index].logmass = logmass;
                    candidates[index].parity_logodds = parity_logodds;
                }
                return;
            }
        }
        candidates[candidate_count] = Candidate4Int{key, logmass, parity_logodds, kIntMetricNegInf};
        candidate_count += 1U;
    };

    for (const State4Int& parent : states) {
        const auto closure_acceptance =
            closure_acceptance_active4_int(parent, column, syndrome, toggle_finite);
        const bool no_accepted = closure_acceptance.first;
        const bool toggle_accepted = closure_acceptance.second;
        if (!no_accepted && !toggle_accepted) {
            continue;
        }
        const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
        const std::int64_t base_logmass = parent.logmass + no_error_log_const;
        Key4 no_key;
        fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);
        if (no_accepted) {
            add_candidate(no_key, base_logmass, parity_pair.first);
        }
        if (toggle_accepted) {
            Key4 toggle_key = no_key;
            toggle_key.logical = logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
            apply_active_toggle_to_child_active4(toggle_key, column);
            add_candidate(toggle_key, base_logmass + toggle_logodds, parity_pair.second);
        }
    }

    if (model.collect_phase_timing) {
        result.stats.transition_time_s += now_seconds() - transition_started;
    }
    result.stats.max_pre_prune_state_count = std::max(
        result.stats.max_pre_prune_state_count,
        static_cast<std::uint64_t>(candidate_count)
    );
    result.stats.sum_pre_prune_state_count += static_cast<std::uint64_t>(candidate_count);
    if (candidate_count == 0) {
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        result.ok = false;
        no_path = true;
        return true;
    }

    const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
    std::int64_t best_score = kIntMetricNegInf;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        candidates[index].score =
            score_int_metric(
                candidates[index].logmass,
                candidates[index].parity_logodds,
                alpha_int,
                int_metric_scale
            );
        if (profile_enabled) result.stats.profile_score_evals += 1ULL;
        if (candidates[index].score > best_score) {
            best_score = candidates[index].score;
        }
    }

    const std::int64_t cutoff = best_score - delta_int;
    std::array<std::size_t, kActive4SmallStateStepLimit * 2U> survivor_indices{};
    std::size_t survivor_count = 0;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        if (candidates[index].score >= cutoff) {
            survivor_indices[survivor_count] = index;
            survivor_count += 1U;
        }
    }

    Active4CandidateIntBetter better{&column, model.n_logical_limbs};
    auto better_index = [&](std::size_t lhs, std::size_t rhs) {
        return better(candidates[lhs], candidates[rhs]);
    };
    if (survivor_count > static_cast<std::size_t>(K)) {
        std::sort(
            survivor_indices.begin(),
            survivor_indices.begin() + static_cast<std::ptrdiff_t>(survivor_count),
            better_index
        );
        survivor_count = static_cast<std::size_t>(K);
    } else if (!disable_final_prune_sort) {
        std::sort(
            survivor_indices.begin(),
            survivor_indices.begin() + static_cast<std::ptrdiff_t>(survivor_count),
            better_index
        );
    }

    states.clear();
    states.reserve(survivor_count);
    for (std::size_t index = 0; index < survivor_count; ++index) {
        const Candidate4Int& survivor = candidates[survivor_indices[index]];
        states.push_back(State4Int{survivor.key, survivor.logmass, survivor.parity_logodds});
    }
    if (model.collect_phase_timing) {
        result.stats.prune_time_s += now_seconds() - prune_started;
    }

    const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
    result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
    result.stats.sum_post_prune_state_count += post_count;
    if (states.empty()) {
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        result.ok = false;
        no_path = true;
    }
    return true;
}

DecodeResult decode_native_frontier_lite_active4_with_workspace(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    int int_metric_scale,
    MaxLogIntWorkspace& workspace
) {
    const double total_started = now_seconds();
    DecodeResult result;
    result.stats.no_path_count = 0;

    const std::int64_t delta_int = std::max<std::int64_t>(0, quantize_metric(Delta, int_metric_scale));
    const std::int64_t alpha_int = quantize_metric(score_alpha, int_metric_scale);

    std::vector<State4Int>& states = workspace.states4;
    std::vector<Candidate4Int>& merged = workspace.merged4;
    std::vector<std::size_t>& survivor_indices = workspace.survivor_indices;
    std::vector<std::size_t>& merge_bucket_indices = workspace.merge_bucket_indices;
    std::vector<std::uint32_t>& merge_bucket_generations = workspace.merge_bucket_generations;
    std::uint32_t& merge_generation = workspace.merge_generation;
    const bool disable_one_pass_prune = one_pass_prune_disabled();
    const std::size_t one_pass_prune_min = one_pass_prune_min_candidates();
    const bool disable_final_prune_sort = final_prune_sort_disabled();
    const bool disable_close_empty_split_merge = close_empty_split_merge_disabled();
    const bool disable_no_merge_transition = no_merge_transition_disabled();
    const bool disable_single_parent_step = single_parent_step_disabled();
    const bool disable_small_state_step = small_state_step_disabled();
    const bool use_single_parent_step =
        !disable_single_parent_step && !model.force_full_key && model.n_limbs > kCompactLimbs && model.active4_supported;
    const bool use_small_state_step =
        !disable_small_state_step && use_single_parent_step;
    const bool profile_enabled = native_profile_enabled();

    states.clear();
    states.push_back(State4Int{Key4{}, 0, 0});

    for (std::size_t column_index = 0; column_index < model.columns.size(); ++column_index) {
        const Column& column = model.columns[column_index];
        result.stats.processed_columns = static_cast<int>(column_index) + 1;
        merged.clear();
        merged.reserve(states.size() * 2);
        std::size_t merge_bucket_mask = 0;
        bool use_merge_index = false;
        const bool toggle_finite = std::isfinite(column.toggle_logodds);
        const bool close_empty = column.close_nonzero_count == 0;
        const std::int64_t no_error_log_const = column_no_error_log_const_int(column, int_metric_scale);
        const std::int64_t toggle_logodds = column_toggle_logodds_int(column, int_metric_scale);

        auto merge_existing_max = [&](Candidate4Int& existing, std::int64_t logmass, std::int64_t parity_logodds) {
            if (logmass > existing.logmass) {
                existing.logmass = logmass;
                existing.parity_logodds = parity_logodds;
            }
        };

        auto emit_child = [&](const Key4& key, std::int64_t logmass, std::int64_t parity_logodds) {
            if (profile_enabled) result.stats.profile_emit_child_calls += 1ULL;
            if (!use_merge_index) {
                for (Candidate4Int& existing : merged) {
                    if (active4_key_equal_after(existing.key, key, column, model.n_logical_limbs)) {
                        merge_existing_max(existing, logmass, parity_logodds);
                        if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                        return;
                    }
                }
                if (merged.size() < kLinearMergeLimit) {
                    merged.push_back(Candidate4Int{key, logmass, parity_logodds, kIntMetricNegInf});
                    return;
                }
                const std::size_t bucket_count = merge_bucket_count_for(states.size() * 2 + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < merged.size(); ++index) {
                    std::size_t slot =
                        active4_hash_after(merged[index].key, column, model.n_logical_limbs) & merge_bucket_mask;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                use_merge_index = true;
            }
            std::size_t slot = active4_hash_after(key, column, model.n_logical_limbs) & merge_bucket_mask;
            std::uint64_t probes = 1ULL;
            while (true) {
                if (merge_bucket_generations[slot] != merge_generation) {
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    const std::size_t index = merged.size();
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                    merged.push_back(Candidate4Int{key, logmass, parity_logodds, kIntMetricNegInf});
                    return;
                }
                const std::size_t existing_index = merge_bucket_indices[slot];
                Candidate4Int& existing = merged[existing_index];
                if (active4_key_equal_after(existing.key, key, column, model.n_logical_limbs)) {
                    merge_existing_max(existing, logmass, parity_logodds);
                    if (profile_enabled) {
                        result.stats.profile_merge_duplicate_count += 1ULL;
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    return;
                }
                slot = (slot + 1U) & merge_bucket_mask;
                probes += 1ULL;
            }
        };

        const double transition_started = model.collect_phase_timing ? now_seconds() : 0.0;
        result.stats.transition_evals +=
            static_cast<std::uint64_t>(states.size()) * static_cast<std::uint64_t>(toggle_finite ? 2 : 1);
        if (
            use_small_state_step
            && states.size() >= 2
            && states.size() <= kActive4SmallStateStepLimit
        ) {
            bool no_path = false;
            if (try_active4_small_state_step_int(
                    model,
                    column,
                    syndrome,
                    K,
                    delta_int,
                    alpha_int,
                    int_metric_scale,
                    disable_final_prune_sort,
                    profile_enabled,
                    transition_started,
                    total_started,
                    states,
                    result,
                    no_path
                )) {
                if (no_path) {
                    return result;
                }
                continue;
            }
        }
        if (use_single_parent_step && states.size() == 1) {
            const State4Int parent = states[0];
            const auto closure_acceptance =
                closure_acceptance_active4_int(parent, column, syndrome, toggle_finite);
            const bool no_accepted = closure_acceptance.first;
            const bool toggle_accepted = closure_acceptance.second;
            Candidate4Int first_candidate;
            Candidate4Int second_candidate;
            bool have_first_candidate = false;
            bool have_second_candidate = false;
            auto add_direct_candidate = [&](const Key4& key, std::int64_t logmass, std::int64_t parity_logodds) {
                if (!have_first_candidate) {
                    first_candidate = Candidate4Int{key, logmass, parity_logodds, kIntMetricNegInf};
                    have_first_candidate = true;
                    return;
                }
                if (active4_key_equal_after(first_candidate.key, key, column, model.n_logical_limbs)) {
                    if (logmass > first_candidate.logmass) {
                        first_candidate.logmass = logmass;
                        first_candidate.parity_logodds = parity_logodds;
                    }
                    return;
                }
                second_candidate = Candidate4Int{key, logmass, parity_logodds, kIntMetricNegInf};
                have_second_candidate = true;
            };

            if (no_accepted || toggle_accepted) {
                const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                const std::int64_t base_logmass = parent.logmass + no_error_log_const;
                Key4 no_key;
                if (no_accepted || toggle_accepted) {
                    fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);
                }
                if (no_accepted) {
                    add_direct_candidate(no_key, base_logmass, parity_pair.first);
                }
                if (toggle_accepted) {
                    Key4 toggle_key = no_key;
                    toggle_key.logical = logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                    add_direct_candidate(toggle_key, base_logmass + toggle_logodds, parity_pair.second);
                }
            }
            if (model.collect_phase_timing) {
                result.stats.transition_time_s += now_seconds() - transition_started;
            }

            const std::uint64_t candidate_count =
                static_cast<std::uint64_t>(have_first_candidate ? (have_second_candidate ? 2 : 1) : 0);
            result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
            result.stats.sum_pre_prune_state_count += candidate_count;
            if (candidate_count == 0) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }

            const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
            first_candidate.score =
                score_int_metric(
                    first_candidate.logmass,
                    first_candidate.parity_logodds,
                    alpha_int,
                    int_metric_scale
                );
            if (profile_enabled) result.stats.profile_score_evals += 1ULL;
            std::int64_t best_score = first_candidate.score;
            if (have_second_candidate) {
                second_candidate.score =
                    score_int_metric(
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                        alpha_int,
                        int_metric_scale
                    );
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (second_candidate.score > best_score) best_score = second_candidate.score;
            }
            const std::int64_t cutoff = best_score - delta_int;
            const bool keep_first = first_candidate.score >= cutoff;
            const bool keep_second = have_second_candidate && second_candidate.score >= cutoff;

            states.clear();
            Active4CandidateIntBetter better{&column, model.n_logical_limbs};
            if (keep_first && keep_second && K == 1) {
                const Candidate4Int& survivor =
                    better(second_candidate, first_candidate) ? second_candidate : first_candidate;
                states.push_back(State4Int{survivor.key, survivor.logmass, survivor.parity_logodds});
            } else if (keep_first && keep_second && !disable_final_prune_sort) {
                if (better(second_candidate, first_candidate)) {
                    states.push_back(State4Int{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                    states.push_back(State4Int{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                } else {
                    states.push_back(State4Int{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                    states.push_back(State4Int{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            } else {
                if (keep_first) {
                    states.push_back(State4Int{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                }
                if (keep_second) {
                    states.push_back(State4Int{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            }
            if (model.collect_phase_timing) {
                result.stats.prune_time_s += now_seconds() - prune_started;
            }

            const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
            result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
            result.stats.sum_post_prune_state_count += post_count;
            if (states.empty()) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }
            continue;
        }
        if (
            close_empty && toggle_finite && column.no_child_injective4 && column.toggle_has_new_active_bit
                && !disable_no_merge_transition
        ) {
            if (profile_enabled) result.stats.profile_no_merge_transition_columns += 1ULL;
            merged.reserve(states.size() * 2);
            for (const State4Int& parent : states) {
                const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                Key4 no_key;
                fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);

                const std::int64_t base_logmass = parent.logmass + no_error_log_const;
                merged.push_back(Candidate4Int{
                    no_key,
                    base_logmass,
                    parity_pair.first,
                    kIntMetricNegInf,
                });

                Key4 toggle_key = no_key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child_active4(toggle_key, column);
                merged.push_back(Candidate4Int{
                    toggle_key,
                    base_logmass + toggle_logodds,
                    parity_pair.second,
                    kIntMetricNegInf,
                });
            }
        } else if (
            close_empty && toggle_finite && column.no_child_injective4 && !disable_close_empty_split_merge
        ) {
            if (profile_enabled) result.stats.profile_split_merge_columns += 1ULL;
            for (const State4Int& parent : states) {
                const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                Key4 no_key;
                fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);

                const std::int64_t base_logmass = parent.logmass + no_error_log_const;
                merged.push_back(Candidate4Int{
                    no_key,
                    base_logmass,
                    parity_pair.first,
                    kIntMetricNegInf,
                });
            }

            const std::size_t no_count = merged.size();
            if (no_count < kLinearMergeLimit) {
                for (const State4Int& parent : states) {
                    const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                    Key4 toggle_key;
                    fill_no_child_key_active4_int(toggle_key, parent, column, parent.key.logical);
                    logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                    const Candidate4Int toggle_candidate{
                        toggle_key,
                        parent.logmass + no_error_log_const + toggle_logodds,
                        parity_pair.second,
                        kIntMetricNegInf,
                    };
                    bool matched = false;
                    for (std::size_t no_index = 0; no_index < merged.size(); ++no_index) {
                        Candidate4Int& existing = merged[no_index];
                        if (active4_key_equal_after(
                                existing.key,
                                toggle_candidate.key,
                                column,
                                model.n_logical_limbs
                            )) {
                            merge_existing_max(
                                existing,
                                toggle_candidate.logmass,
                                toggle_candidate.parity_logodds
                            );
                            if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        merged.push_back(toggle_candidate);
                    }
                }
            } else {
                const std::size_t bucket_count = merge_bucket_count_for(no_count + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < no_count; ++index) {
                    std::size_t slot =
                        active4_hash_after(merged[index].key, column, model.n_logical_limbs) & merge_bucket_mask;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                for (const State4Int& parent : states) {
                    const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                    Key4 toggle_key;
                    fill_no_child_key_active4_int(toggle_key, parent, column, parent.key.logical);
                    logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                    const Candidate4Int toggle_candidate{
                        toggle_key,
                        parent.logmass + no_error_log_const + toggle_logodds,
                        parity_pair.second,
                        kIntMetricNegInf,
                    };
                    std::size_t slot =
                        active4_hash_after(toggle_candidate.key, column, model.n_logical_limbs) & merge_bucket_mask;
                    bool matched = false;
                    std::uint64_t probes = 1ULL;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        const std::size_t existing_index = merge_bucket_indices[slot];
                        Candidate4Int& existing = merged[existing_index];
                        if (active4_key_equal_after(
                                existing.key,
                                toggle_candidate.key,
                                column,
                                model.n_logical_limbs
                            )) {
                            merge_existing_max(
                                existing,
                                toggle_candidate.logmass,
                                toggle_candidate.parity_logodds
                            );
                            if (profile_enabled) result.stats.profile_merge_duplicate_count += 1ULL;
                            matched = true;
                            break;
                        }
                        slot = (slot + 1U) & merge_bucket_mask;
                        probes += 1ULL;
                    }
                    if (profile_enabled) {
                        result.stats.profile_hash_probe_total += probes;
                        result.stats.profile_hash_probe_max =
                            std::max(result.stats.profile_hash_probe_max, probes);
                    }
                    if (!matched) {
                        const std::size_t index = merged.size();
                        merge_bucket_generations[slot] = merge_generation;
                        merge_bucket_indices[slot] = index;
                        merged.push_back(toggle_candidate);
                    }
                }
            }
        } else if (close_empty && toggle_finite) {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4Int& parent : states) {
                const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                Key4 no_key;
                fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);
                Key4 toggle_key = no_key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child_active4(toggle_key, column);

                const std::int64_t base_logmass = parent.logmass + no_error_log_const;
                emit_child(no_key, base_logmass, parity_pair.first);
                emit_child(toggle_key, base_logmass + toggle_logodds, parity_pair.second);
            }
        } else if (close_empty) {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4Int& parent : states) {
                const auto parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                Key4 no_key;
                fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);
                emit_child(no_key, parent.logmass + no_error_log_const, parity_pair.first);
            }
        } else {
            if (profile_enabled) result.stats.profile_generic_merge_columns += 1ULL;
            for (const State4Int& parent : states) {
                const auto closure_acceptance =
                    closure_acceptance_active4_int(parent, column, syndrome, toggle_finite);
                const bool no_accepted = closure_acceptance.first;
                const bool toggle_accepted = closure_acceptance.second;

                Key4 no_key;
                Key4 toggle_key;
                if (no_accepted || toggle_accepted) {
                    fill_no_child_key_active4_int(no_key, parent, column, parent.key.logical);
                }
                if (toggle_accepted) {
                    toggle_key = no_key;
                    toggle_key.logical = logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
                    apply_active_toggle_to_child_active4(toggle_key, column);
                }

                std::pair<std::int64_t, std::int64_t> parity_pair;
                if (no_accepted || toggle_accepted) {
                    parity_pair = child_parities_active4_int(parent, column, syndrome, int_metric_scale);
                }
                if (no_accepted) {
                    emit_child(no_key, parent.logmass + no_error_log_const, parity_pair.first);
                }
                if (toggle_accepted) {
                    emit_child(toggle_key, parent.logmass + no_error_log_const + toggle_logodds, parity_pair.second);
                }
            }
        }
        if (model.collect_phase_timing) {
            result.stats.transition_time_s += now_seconds() - transition_started;
        }

        const std::uint64_t candidate_count = static_cast<std::uint64_t>(merged.size());
        result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
        result.stats.sum_pre_prune_state_count += candidate_count;
        if (merged.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }

        const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
        std::int64_t best_score = kIntMetricNegInf;
        survivor_indices.clear();
        const bool use_one_pass_prune =
            !disable_one_pass_prune && merged.size() >= one_pass_prune_min;
        if (!use_one_pass_prune) {
            for (Candidate4Int& candidate : merged) {
                candidate.score =
                    score_int_metric(candidate.logmass, candidate.parity_logodds, alpha_int, int_metric_scale);
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (candidate.score > best_score) best_score = candidate.score;
            }
            const std::int64_t cutoff = best_score - delta_int;
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                if (merged[index].score >= cutoff) survivor_indices.push_back(index);
            }
        } else {
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                Candidate4Int& candidate = merged[index];
                candidate.score =
                    score_int_metric(candidate.logmass, candidate.parity_logodds, alpha_int, int_metric_scale);
                if (profile_enabled) result.stats.profile_score_evals += 1ULL;
                if (candidate.score > best_score) best_score = candidate.score;
                if (candidate.score >= best_score - delta_int) {
                    survivor_indices.push_back(index);
                }
            }
            const std::int64_t cutoff = best_score - delta_int;
            std::size_t write_index = 0;
            for (std::size_t read_index = 0; read_index < survivor_indices.size(); ++read_index) {
                const std::size_t candidate_index = survivor_indices[read_index];
                if (merged[candidate_index].score >= cutoff) {
                    survivor_indices[write_index] = candidate_index;
                    write_index += 1U;
                }
            }
            survivor_indices.resize(write_index);
        }
        Active4CandidateIntBetter better{&column, model.n_logical_limbs};
        auto better_index = [&](std::size_t lhs, std::size_t rhs) {
            return better(merged[lhs], merged[rhs]);
        };
        if (static_cast<int>(survivor_indices.size()) > K) {
            if (profile_enabled) result.stats.profile_nth_element_calls += 1ULL;
            std::nth_element(
                survivor_indices.begin(),
                survivor_indices.begin() + K,
                survivor_indices.end(),
                better_index
            );
            survivor_indices.resize(static_cast<std::size_t>(K));
        }
        if (!disable_final_prune_sort) {
            std::sort(survivor_indices.begin(), survivor_indices.end(), better_index);
        }
        states.clear();
        states.reserve(survivor_indices.size());
        for (std::size_t survivor_index : survivor_indices) {
            const Candidate4Int& survivor = merged[survivor_index];
            states.push_back(State4Int{survivor.key, survivor.logmass, survivor.parity_logodds});
        }
        if (model.collect_phase_timing) {
            result.stats.prune_time_s += now_seconds() - prune_started;
        }

        const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
        result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
        result.stats.sum_post_prune_state_count += post_count;
        if (states.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }
    }

    std::map<LogicalMask, std::int64_t> terminal_masses;
    const Column& terminal_column = model.columns.empty() ? Column{} : model.columns.back();
    for (const State4Int& state : states) {
        bool nonzero = false;
        for (int slot = 0; slot < terminal_column.active_nonzero_count; ++slot) {
            if (state.key.det[static_cast<std::size_t>(slot)] != 0) {
                nonzero = true;
                break;
            }
        }
        if (nonzero) continue;
        auto inserted = terminal_masses.emplace(state.key.logical, state.logmass);
        if (!inserted.second && state.logmass > inserted.first->second) {
            inserted.first->second = state.logmass;
        }
    }

    if (terminal_masses.empty()) {
        result.ok = false;
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        return result;
    }

    result.ok = true;
    std::int64_t top1 = kIntMetricNegInf;
    std::int64_t top2 = kIntMetricNegInf;
    bool have_hat = false;
    for (const auto& item : terminal_masses) {
        const LogicalMask& logical = item.first;
        const std::int64_t logmass_int = item.second;
        result.terminal_log_masses[logical] =
            static_cast<double>(logmass_int) / static_cast<double>(int_metric_scale);
        if (!have_hat || logmass_int > top1 || (logmass_int == top1 && logical < result.logical_hat)) {
            top2 = top1;
            top1 = logmass_int;
            result.logical_hat = logical;
            have_hat = true;
        } else if (logmass_int > top2) {
            top2 = logmass_int;
        }
    }
    result.log_evidence = static_cast<double>(top1) / static_cast<double>(int_metric_scale);
    result.terminal_top_log_mass_gap =
        top2 > kIntMetricNegInf / 2
            ? static_cast<double>(top1 - top2) / static_cast<double>(int_metric_scale)
            : std::numeric_limits<double>::infinity();
    result.stats.no_path_count = 0;
    result.stats.total_time_s = now_seconds() - total_started;
    return result;
}

DecodeResult decode_native_maxlog_int_full_with_workspace(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    int int_metric_scale,
    MaxLogIntWorkspace& workspace
) {
    const double total_started = now_seconds();
    DecodeResult result;
    result.stats.no_path_count = 0;

    const std::int64_t delta_int = std::max<std::int64_t>(0, quantize_metric(Delta, int_metric_scale));
    const std::int64_t alpha_int = quantize_metric(score_alpha, int_metric_scale);
    const bool disable_final_prune_sort = final_prune_sort_disabled();

    std::vector<StateInt>& states = workspace.states;
    std::vector<CandidateInt>& merged = workspace.merged;
    std::vector<std::size_t>& survivor_indices = workspace.survivor_indices;

    states.clear();
    states.push_back(StateInt{Key{}, 0, 0});

    for (std::size_t column_index = 0; column_index < model.columns.size(); ++column_index) {
        const Column& column = model.columns[column_index];
        result.stats.processed_columns = static_cast<int>(column_index) + 1;
        const bool toggle_finite = std::isfinite(column.toggle_logodds);
        const std::int64_t no_error_log_const = column_no_error_log_const_int(column, int_metric_scale);
        const std::int64_t toggle_logodds = column_toggle_logodds_int(column, int_metric_scale);

        const double transition_started = model.collect_phase_timing ? now_seconds() : 0.0;
        std::map<Key, CandidateInt, KeyLess> merged_by_key{KeyLess{model.n_limbs}};
        result.stats.transition_evals +=
            static_cast<std::uint64_t>(states.size()) * static_cast<std::uint64_t>(toggle_finite ? 2 : 1);

        for (const StateInt& parent : states) {
            const auto closure_acceptance =
                closure_acceptance_sparse(parent.key, column, syndrome, toggle_finite);
            const bool no_accepted = closure_acceptance.first;
            const bool toggle_accepted = closure_acceptance.second;
            if (!no_accepted && !toggle_accepted) {
                continue;
            }

            const auto parity_pair = child_parities_int(parent, column, syndrome, int_metric_scale);
            const std::int64_t base_logmass = parent.logmass + no_error_log_const;
            Key no_key;
            Key toggle_key;
            const LogicalMask toggle_logical =
                logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
            if (no_accepted && toggle_accepted) {
                fill_no_child_key(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                toggle_key = no_key;
                toggle_key.logical = toggle_logical;
                apply_active_toggle_to_child(toggle_key, column);
            } else if (no_accepted) {
                fill_no_child_key(no_key, parent.key, column, model.n_limbs, parent.key.logical);
            } else if (toggle_accepted) {
                fill_toggle_child_key(toggle_key, parent.key, column, model.n_limbs, toggle_logical);
            }

            if (no_accepted) {
                merge_candidate_maxlog(
                    merged_by_key,
                    CandidateInt{no_key, base_logmass, parity_pair.first, kIntMetricNegInf}
                );
            }
            if (toggle_accepted) {
                merge_candidate_maxlog(
                    merged_by_key,
                    CandidateInt{
                        toggle_key,
                        base_logmass + toggle_logodds,
                        parity_pair.second,
                        kIntMetricNegInf,
                    }
                );
            }
        }
        if (model.collect_phase_timing) {
            result.stats.transition_time_s += now_seconds() - transition_started;
        }

        const std::uint64_t candidate_count = static_cast<std::uint64_t>(merged_by_key.size());
        result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
        result.stats.sum_pre_prune_state_count += candidate_count;
        if (merged_by_key.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }

        const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
        merged.clear();
        merged.reserve(merged_by_key.size());
        for (auto& item : merged_by_key) {
            merged.push_back(item.second);
        }

        std::int64_t best_score = kIntMetricNegInf;
        for (CandidateInt& candidate : merged) {
            candidate.score = score_int_metric(candidate.logmass, candidate.parity_logodds, alpha_int, int_metric_scale);
            best_score = std::max(best_score, candidate.score);
        }
        const std::int64_t cutoff = best_score - delta_int;
        survivor_indices.clear();
        survivor_indices.reserve(merged.size());
        for (std::size_t index = 0; index < merged.size(); ++index) {
            if (merged[index].score >= cutoff) {
                survivor_indices.push_back(index);
            }
        }

        CandidateIntBetter better{model.n_limbs};
        auto better_index = [&](std::size_t lhs, std::size_t rhs) {
            return better(merged[lhs], merged[rhs]);
        };
        if (static_cast<int>(survivor_indices.size()) > K) {
            std::nth_element(
                survivor_indices.begin(),
                survivor_indices.begin() + K,
                survivor_indices.end(),
                better_index
            );
            survivor_indices.resize(static_cast<std::size_t>(K));
        }
        if (!disable_final_prune_sort) {
            std::sort(survivor_indices.begin(), survivor_indices.end(), better_index);
        }

        states.clear();
        states.reserve(survivor_indices.size());
        for (std::size_t survivor_index : survivor_indices) {
            const CandidateInt& survivor = merged[survivor_index];
            states.push_back(StateInt{survivor.key, survivor.logmass, survivor.parity_logodds});
        }
        if (model.collect_phase_timing) {
            result.stats.prune_time_s += now_seconds() - prune_started;
        }

        const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
        result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
        result.stats.sum_post_prune_state_count += post_count;
        if (states.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }
    }

    std::map<LogicalMask, std::int64_t> terminal_masses;
    for (const StateInt& state : states) {
        if (!det_zero(state.key, model.n_limbs)) continue;
        auto inserted = terminal_masses.emplace(state.key.logical, state.logmass);
        if (!inserted.second && state.logmass > inserted.first->second) {
            inserted.first->second = state.logmass;
        }
    }

    if (terminal_masses.empty()) {
        result.ok = false;
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        return result;
    }

    result.ok = true;
    std::int64_t top1 = kIntMetricNegInf;
    std::int64_t top2 = kIntMetricNegInf;
    bool have_hat = false;
    for (const auto& item : terminal_masses) {
        const LogicalMask& logical = item.first;
        const std::int64_t logmass_int = item.second;
        result.terminal_log_masses[logical] =
            static_cast<double>(logmass_int) / static_cast<double>(int_metric_scale);
        if (!have_hat || logmass_int > top1 || (logmass_int == top1 && logical < result.logical_hat)) {
            top2 = top1;
            top1 = logmass_int;
            result.logical_hat = logical;
            have_hat = true;
        } else if (logmass_int > top2) {
            top2 = logmass_int;
        }
    }
    result.log_evidence = static_cast<double>(top1) / static_cast<double>(int_metric_scale);
    result.terminal_top_log_mass_gap =
        top2 > kIntMetricNegInf / 2
            ? static_cast<double>(top1 - top2) / static_cast<double>(int_metric_scale)
            : std::numeric_limits<double>::infinity();
    result.stats.no_path_count = 0;
    result.stats.total_time_s = now_seconds() - total_started;
    return result;
}

DecodeResult decode_native_full_with_workspace(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    FullWorkspace& workspace
) {
    const double total_started = now_seconds();
    DecodeResult result;
    result.stats.no_path_count = 0;

    std::vector<State>& states = workspace.states;
    std::vector<Candidate>& merged = workspace.merged;
    std::vector<Candidate>& toggle_candidates = workspace.toggle_candidates;
    std::vector<std::size_t>& survivor_indices = workspace.survivor_indices;
    std::vector<std::size_t>& merge_bucket_indices = workspace.merge_bucket_indices;
    std::vector<std::uint32_t>& merge_bucket_generations = workspace.merge_bucket_generations;
    std::uint32_t& merge_generation = workspace.merge_generation;
    const bool disable_one_pass_prune = one_pass_prune_disabled();
    const std::size_t one_pass_prune_min = one_pass_prune_min_candidates();
    const bool disable_final_prune_sort = final_prune_sort_disabled();
    const bool disable_close_empty_split_merge = close_empty_split_merge_disabled();
    const bool disable_no_merge_transition = no_merge_transition_disabled();
    const bool disable_single_parent_step = single_parent_step_disabled();
    const bool use_single_parent_step =
        !disable_single_parent_step && !model.force_full_key && model.n_limbs > kCompactLimbs && model.active4_supported;

    states.clear();
    states.push_back(State{Key{}, 0.0, 0.0});

    for (std::size_t column_index = 0; column_index < model.columns.size(); ++column_index) {
        const Column& column = model.columns[column_index];
        result.stats.processed_columns = static_cast<int>(column_index) + 1;
        merged.clear();
        merged.reserve(states.size() * 2);
        std::size_t merge_bucket_mask = 0;
        bool use_merge_index = false;
        KeyEqual key_equal{model.n_limbs};

        auto emit_child = [&](const Key& key, double logmass, double parity_logodds) {
            if (!use_merge_index) {
                for (Candidate& existing : merged) {
                    if (key_equal(existing.key, key)) {
                        existing.logmass = logaddexp_pair(existing.logmass, logmass);
                        return;
                    }
                }
                if (merged.size() < kLinearMergeLimit) {
                    merged.push_back(Candidate{key, logmass, parity_logodds, -std::numeric_limits<double>::infinity()});
                    return;
                }
                const std::size_t bucket_count = merge_bucket_count_for(states.size() * 2 + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < merged.size(); ++index) {
                    std::size_t slot =
                        hash_key_for_column(
                            merged[index].key,
                            column,
                            model.n_limbs,
                            model.n_logical_limbs
                        ) & merge_bucket_mask;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                use_merge_index = true;
            }
            std::size_t slot =
                hash_key_for_column(key, column, model.n_limbs, model.n_logical_limbs) & merge_bucket_mask;
            while (true) {
                if (merge_bucket_generations[slot] != merge_generation) {
                    const std::size_t index = merged.size();
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                    merged.push_back(Candidate{key, logmass, parity_logodds, -std::numeric_limits<double>::infinity()});
                    return;
                }
                const std::size_t existing_index = merge_bucket_indices[slot];
                Candidate& existing = merged[existing_index];
                if (key_equal_for_column(existing.key, key, column, model.n_limbs)) {
                    existing.logmass = logaddexp_pair(existing.logmass, logmass);
                    return;
                }
                slot = (slot + 1U) & merge_bucket_mask;
            }
        };

        const double transition_started = model.collect_phase_timing ? now_seconds() : 0.0;
        const bool toggle_finite = std::isfinite(column.toggle_logodds);
        const bool close_empty = column.close_nonzero_count == 0;
        result.stats.transition_evals +=
            static_cast<std::uint64_t>(states.size()) * static_cast<std::uint64_t>(toggle_finite ? 2 : 1);
        if (use_single_parent_step && states.size() == 1) {
            const State parent = states[0];
            const auto closure_acceptance =
                closure_acceptance_sparse(parent.key, column, syndrome, toggle_finite);
            const bool no_accepted = closure_acceptance.first;
            const bool toggle_accepted = closure_acceptance.second;
            Candidate first_candidate;
            Candidate second_candidate;
            bool have_first_candidate = false;
            bool have_second_candidate = false;
            auto add_direct_candidate = [&](const Key& key, double logmass, double parity_logodds) {
                if (!have_first_candidate) {
                    first_candidate = Candidate{
                        key,
                        logmass,
                        parity_logodds,
                        -std::numeric_limits<double>::infinity(),
                    };
                    have_first_candidate = true;
                    return;
                }
                if (KeyEqual{model.n_limbs}(first_candidate.key, key)) {
                    first_candidate.logmass = logaddexp_pair(first_candidate.logmass, logmass);
                    return;
                }
                second_candidate = Candidate{
                    key,
                    logmass,
                    parity_logodds,
                    -std::numeric_limits<double>::infinity(),
                };
                have_second_candidate = true;
            };

            if (no_accepted || toggle_accepted) {
                const auto parity_pair = child_parities_fast(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                if (no_accepted) {
                    Key no_key;
                    fill_no_child_key(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                    add_direct_candidate(no_key, base_logmass, parity_pair.first);
                }
                if (toggle_accepted) {
                    Key toggle_key;
                    fill_toggle_child_key(
                        toggle_key,
                        parent.key,
                        column,
                        model.n_limbs,
                        logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs)
                    );
                    add_direct_candidate(
                        toggle_key,
                        base_logmass + column.toggle_logodds,
                        parity_pair.second
                    );
                }
            }
            if (model.collect_phase_timing) {
                result.stats.transition_time_s += now_seconds() - transition_started;
            }

            const std::uint64_t candidate_count =
                static_cast<std::uint64_t>(have_first_candidate ? (have_second_candidate ? 2 : 1) : 0);
            result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
            result.stats.sum_pre_prune_state_count += candidate_count;
            if (candidate_count == 0) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }

            const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
            first_candidate.score = first_candidate.logmass + score_alpha * first_candidate.parity_logodds;
            double best_score = first_candidate.score;
            if (have_second_candidate) {
                second_candidate.score = second_candidate.logmass + score_alpha * second_candidate.parity_logodds;
                if (second_candidate.score > best_score) best_score = second_candidate.score;
            }
            const double cutoff = best_score - Delta;
            const bool keep_first = first_candidate.score >= cutoff;
            const bool keep_second = have_second_candidate && second_candidate.score >= cutoff;

            states.clear();
            CandidateBetter better{model.n_limbs};
            if (keep_first && keep_second && K == 1) {
                const Candidate& survivor =
                    better(second_candidate, first_candidate) ? second_candidate : first_candidate;
                states.push_back(State{survivor.key, survivor.logmass, survivor.parity_logodds});
            } else if (keep_first && keep_second && !disable_final_prune_sort) {
                if (better(second_candidate, first_candidate)) {
                    states.push_back(State{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                    states.push_back(State{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                } else {
                    states.push_back(State{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                    states.push_back(State{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            } else {
                if (keep_first) {
                    states.push_back(State{
                        first_candidate.key,
                        first_candidate.logmass,
                        first_candidate.parity_logodds,
                    });
                }
                if (keep_second) {
                    states.push_back(State{
                        second_candidate.key,
                        second_candidate.logmass,
                        second_candidate.parity_logodds,
                    });
                }
            }
            if (model.collect_phase_timing) {
                result.stats.prune_time_s += now_seconds() - prune_started;
            }

            const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
            result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
            result.stats.sum_post_prune_state_count += post_count;
            if (states.empty()) {
                result.stats.no_path_count = 1;
                result.stats.total_time_s = now_seconds() - total_started;
                result.ok = false;
                return result;
            }
            continue;
        }
        if (
            close_empty && toggle_finite && column.no_child_injective4 && column.toggle_has_new_active_bit
                && !disable_no_merge_transition
        ) {
            merged.reserve(states.size() * 2);
            for (const State& parent : states) {
                const auto parity_pair = child_parities_fast(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                merged.push_back(Candidate{
                    parent.key,
                    base_logmass,
                    parity_pair.first,
                    -std::numeric_limits<double>::infinity(),
                });

                Key toggle_key = parent.key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child(toggle_key, column);
                merged.push_back(Candidate{
                    toggle_key,
                    base_logmass + column.toggle_logodds,
                    parity_pair.second,
                    -std::numeric_limits<double>::infinity(),
                });
            }
        } else if (
            close_empty && toggle_finite && column.no_child_injective4 && !disable_close_empty_split_merge
        ) {
            toggle_candidates.clear();
            toggle_candidates.reserve(states.size());
            for (const State& parent : states) {
                const auto parity_pair = child_parities_fast(parent, column, syndrome);
                const double base_logmass = parent.logmass + column.no_error_log_const;
                merged.push_back(Candidate{
                    parent.key,
                    base_logmass,
                    parity_pair.first,
                    -std::numeric_limits<double>::infinity(),
                });

                Key toggle_key = parent.key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child(toggle_key, column);
                toggle_candidates.push_back(Candidate{
                    toggle_key,
                    base_logmass + column.toggle_logodds,
                    parity_pair.second,
                    -std::numeric_limits<double>::infinity(),
                });
            }

            const std::size_t no_count = merged.size();
            if (no_count < kLinearMergeLimit) {
                for (const Candidate& toggle_candidate : toggle_candidates) {
                    bool matched = false;
                    for (std::size_t no_index = 0; no_index < merged.size(); ++no_index) {
                        Candidate& existing = merged[no_index];
                        if (key_equal(existing.key, toggle_candidate.key)) {
                            existing.logmass = logaddexp_pair(existing.logmass, toggle_candidate.logmass);
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        merged.push_back(toggle_candidate);
                    }
                }
            } else {
                const std::size_t bucket_count = merge_bucket_count_for(no_count + 2);
                if (merge_bucket_indices.size() < bucket_count) {
                    merge_bucket_indices.resize(bucket_count, 0);
                    merge_bucket_generations.resize(bucket_count, 0);
                }
                merge_generation += 1U;
                if (merge_generation == 0) {
                    std::fill(merge_bucket_generations.begin(), merge_bucket_generations.end(), 0);
                    merge_generation = 1;
                }
                merge_bucket_mask = bucket_count - 1U;
                for (std::size_t index = 0; index < no_count; ++index) {
                    std::size_t slot =
                        hash_key_for_column(
                            merged[index].key,
                            column,
                            model.n_limbs,
                            model.n_logical_limbs
                        ) & merge_bucket_mask;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        slot = (slot + 1U) & merge_bucket_mask;
                    }
                    merge_bucket_generations[slot] = merge_generation;
                    merge_bucket_indices[slot] = index;
                }
                for (const Candidate& toggle_candidate : toggle_candidates) {
                    std::size_t slot =
                        hash_key_for_column(
                            toggle_candidate.key,
                            column,
                            model.n_limbs,
                            model.n_logical_limbs
                        ) & merge_bucket_mask;
                    bool matched = false;
                    while (merge_bucket_generations[slot] == merge_generation) {
                        const std::size_t existing_index = merge_bucket_indices[slot];
                        Candidate& existing = merged[existing_index];
                        if (key_equal_for_column(existing.key, toggle_candidate.key, column, model.n_limbs)) {
                            existing.logmass = logaddexp_pair(existing.logmass, toggle_candidate.logmass);
                            matched = true;
                            break;
                        }
                        slot = (slot + 1U) & merge_bucket_mask;
                    }
                    if (!matched) {
                        const std::size_t index = merged.size();
                        merge_bucket_generations[slot] = merge_generation;
                        merge_bucket_indices[slot] = index;
                        merged.push_back(toggle_candidate);
                    }
                }
            }
        } else if (close_empty && toggle_finite) {
            for (const State& parent : states) {
                const auto parity_pair = child_parities_fast(parent, column, syndrome);
                Key toggle_key = parent.key;
                logical_xor_inplace(toggle_key.logical, column.toggle_logical, model.n_logical_limbs);
                apply_active_toggle_to_child(toggle_key, column);

                const double base_logmass = parent.logmass + column.no_error_log_const;
                emit_child(parent.key, base_logmass, parity_pair.first);
                emit_child(
                    toggle_key,
                    base_logmass + column.toggle_logodds,
                    parity_pair.second
                );
            }
        } else if (close_empty) {
            for (const State& parent : states) {
                const auto parity_pair = child_parities_fast(parent, column, syndrome);
                emit_child(parent.key, parent.logmass + column.no_error_log_const, parity_pair.first);
            }
        } else {
            for (const State& parent : states) {
                const auto closure_acceptance =
                    closure_acceptance_sparse(parent.key, column, syndrome, toggle_finite);
                const bool no_accepted = closure_acceptance.first;
                const bool toggle_accepted = closure_acceptance.second;

                Key no_key;
                Key toggle_key;
                const LogicalMask toggle_logical =
                    logical_xor(parent.key.logical, column.toggle_logical, model.n_logical_limbs);
                if (no_accepted && toggle_accepted) {
                    fill_no_child_key(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                    toggle_key = no_key;
                    toggle_key.logical = toggle_logical;
                    apply_active_toggle_to_child(toggle_key, column);
                } else if (no_accepted) {
                    fill_no_child_key(no_key, parent.key, column, model.n_limbs, parent.key.logical);
                } else if (toggle_accepted) {
                    fill_toggle_child_key(toggle_key, parent.key, column, model.n_limbs, toggle_logical);
                }

                std::pair<double, double> parity_pair;
                if (no_accepted || toggle_accepted) {
                    parity_pair = child_parities_fast(parent, column, syndrome);
                }

                if (no_accepted) {
                    emit_child(no_key, parent.logmass + column.no_error_log_const, parity_pair.first);
                }

                if (toggle_accepted) {
                    emit_child(
                        toggle_key,
                        parent.logmass + column.no_error_log_const + column.toggle_logodds,
                        parity_pair.second
                    );
                }
            }
        }
        if (model.collect_phase_timing) {
            result.stats.transition_time_s += now_seconds() - transition_started;
        }

        const std::uint64_t candidate_count = static_cast<std::uint64_t>(merged.size());
        result.stats.max_pre_prune_state_count = std::max(result.stats.max_pre_prune_state_count, candidate_count);
        result.stats.sum_pre_prune_state_count += candidate_count;
        if (merged.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }

        const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
        double best_score = -std::numeric_limits<double>::infinity();
        survivor_indices.clear();
        const bool use_one_pass_prune =
            !disable_one_pass_prune && merged.size() >= one_pass_prune_min;
        if (!use_one_pass_prune) {
            for (Candidate& candidate : merged) {
                candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
                if (candidate.score > best_score) best_score = candidate.score;
            }
            const double cutoff = best_score - Delta;
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                if (merged[index].score >= cutoff) {
                    survivor_indices.push_back(index);
                }
            }
        } else {
            survivor_indices.reserve(merged.size());
            for (std::size_t index = 0; index < merged.size(); ++index) {
                Candidate& candidate = merged[index];
                candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
                if (candidate.score > best_score) best_score = candidate.score;
                const double running_cutoff = best_score - Delta;
                if (candidate.score >= running_cutoff) {
                    survivor_indices.push_back(index);
                }
            }
            const double cutoff = best_score - Delta;
            std::size_t write_index = 0;
            for (std::size_t read_index = 0; read_index < survivor_indices.size(); ++read_index) {
                const std::size_t candidate_index = survivor_indices[read_index];
                if (merged[candidate_index].score >= cutoff) {
                    survivor_indices[write_index] = candidate_index;
                    write_index += 1U;
                }
            }
            survivor_indices.resize(write_index);
        }
        CandidateBetter better{model.n_limbs};
        auto better_index = [&](std::size_t lhs, std::size_t rhs) {
            return better(merged[lhs], merged[rhs]);
        };
        if (static_cast<int>(survivor_indices.size()) > K) {
            std::nth_element(
                survivor_indices.begin(),
                survivor_indices.begin() + K,
                survivor_indices.end(),
                better_index
            );
            survivor_indices.resize(static_cast<std::size_t>(K));
        }
        if (!disable_final_prune_sort) {
            std::sort(survivor_indices.begin(), survivor_indices.end(), better_index);
        }
        states.clear();
        states.reserve(survivor_indices.size());
        for (std::size_t survivor_index : survivor_indices) {
            const Candidate& survivor = merged[survivor_index];
            states.push_back(State{survivor.key, survivor.logmass, survivor.parity_logodds});
        }
        if (model.collect_phase_timing) {
            result.stats.prune_time_s += now_seconds() - prune_started;
        }

        const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
        result.stats.max_post_prune_state_count = std::max(result.stats.max_post_prune_state_count, post_count);
        result.stats.sum_post_prune_state_count += post_count;
        if (states.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }
    }

    for (const State& state : states) {
        if (!det_zero(state.key, model.n_limbs)) continue;
        auto found = result.terminal_log_masses.find(state.key.logical);
        if (found == result.terminal_log_masses.end()) {
            result.terminal_log_masses[state.key.logical] = state.logmass;
        } else {
            found->second = logaddexp_pair(found->second, state.logmass);
        }
    }

    if (result.terminal_log_masses.empty()) {
        result.ok = false;
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        return result;
    }

    result.ok = true;
    double top1 = -std::numeric_limits<double>::infinity();
    double top2 = -std::numeric_limits<double>::infinity();
    bool have_hat = false;
    result.log_evidence = -std::numeric_limits<double>::infinity();
    for (const auto& item : result.terminal_log_masses) {
        const LogicalMask& logical = item.first;
        const double logmass = item.second;
        result.log_evidence = logaddexp_pair(result.log_evidence, logmass);
        if (!have_hat || logmass > top1 || (logmass == top1 && logical < result.logical_hat)) {
            top2 = top1;
            top1 = logmass;
            result.logical_hat = logical;
            have_hat = true;
        } else if (logmass > top2) {
            top2 = logmass;
        }
    }
    result.terminal_top_log_mass_gap =
        std::isfinite(top2) ? top1 - top2 : std::numeric_limits<double>::infinity();
    result.stats.no_path_count = 0;
    result.stats.total_time_s = now_seconds() - total_started;
    return result;
}

DecodeResult decode_native_full(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha
) {
    FullWorkspace workspace;
    return decode_native_full_with_workspace(model, syndrome, K, Delta, score_alpha, workspace);
}

DecodeResult decode_native_with_workspace(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    Compact4Workspace& compact_workspace,
    FullWorkspace& full_workspace,
    MaxLogIntWorkspace& int_workspace,
    MetricMode metric_mode,
    int int_metric_scale
) {
    if (metric_mode == MetricMode::MaxLogInt) {
        if (!model.force_full_key && model.active4_supported) {
            return decode_native_frontier_lite_active4_with_workspace(
                model,
                syndrome,
                K,
                Delta,
                score_alpha,
                int_metric_scale,
                int_workspace
            );
        }
        return decode_native_maxlog_int_full_with_workspace(
            model,
            syndrome,
            K,
            Delta,
            score_alpha,
            int_metric_scale,
            int_workspace
        );
    }
    if (!model.force_full_key && model.n_limbs <= kCompactLimbs) {
        return decode_native_compact4_with_workspace(model, syndrome, K, Delta, score_alpha, compact_workspace);
    }
    if (!model.force_full_key && model.active4_supported) {
        return decode_native_active4_with_workspace(model, syndrome, K, Delta, score_alpha, compact_workspace);
    }
    return decode_native_full_with_workspace(model, syndrome, K, Delta, score_alpha, full_workspace);
}

DecodeResult decode_native(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    MetricMode metric_mode = MetricMode::LogSumExpFloat,
    int int_metric_scale = 1024
) {
    if (K <= 0) {
        throw std::runtime_error("K must be positive");
    }
    if (Delta < 0.0) {
        throw std::runtime_error("Delta must be non-negative");
    }
    if (!std::isfinite(score_alpha) || score_alpha < 0.0) {
        throw std::runtime_error("score_alpha must be finite and non-negative");
    }
    validate_metric_options(metric_mode, int_metric_scale);
    if (metric_mode == MetricMode::MaxLogInt) {
        MaxLogIntWorkspace workspace;
        if (!model.force_full_key && model.active4_supported) {
            return decode_native_frontier_lite_active4_with_workspace(
                model,
                syndrome,
                K,
                Delta,
                score_alpha,
                int_metric_scale,
                workspace
            );
        }
        return decode_native_maxlog_int_full_with_workspace(
            model,
            syndrome,
            K,
            Delta,
            score_alpha,
            int_metric_scale,
            workspace
        );
    }
    if (!model.force_full_key && model.n_limbs <= kCompactLimbs) {
        return decode_native_compact4(model, syndrome, K, Delta, score_alpha);
    }
    if (!model.force_full_key && model.active4_supported) {
        Compact4Workspace workspace;
        return decode_native_active4_with_workspace(model, syndrome, K, Delta, score_alpha, workspace);
    }
    return decode_native_full(model, syndrome, K, Delta, score_alpha);
}

FirstStageDecodePayload decode_overlap1_first_stage_native(
    const NativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha,
    MetricMode metric_mode,
    int int_metric_scale,
    int max_candidate_cols
) {
    if (K <= 0) {
        throw std::runtime_error("K must be positive");
    }
    if (Delta < 0.0) {
        throw std::runtime_error("Delta must be non-negative");
    }
    if (!std::isfinite(score_alpha) || score_alpha < 0.0) {
        throw std::runtime_error("score_alpha must be finite and non-negative");
    }
    validate_metric_options(metric_mode, int_metric_scale);

    FirstStageDecodePayload payload;
    const double build_started = now_seconds();
    OverlapFirstStageModel reduced = build_overlap1_first_stage_model(model, syndrome);
    payload.build_time_s = now_seconds() - build_started;
    payload.candidate_cols = reduced.candidate_cols;
    payload.reduced_rows = reduced.reduced_rows;
    payload.active_syndrome_weight = reduced.active_syndrome_weight;
    payload.uncovered_active_rows = reduced.uncovered_active_rows;

    if (payload.uncovered_active_rows > 0) {
        payload.status_override = "no_path_gate_uncovered";
        payload.result.ok = false;
        payload.result.stats.no_path_count = 1;
        return payload;
    }
    if (max_candidate_cols > 0 && payload.candidate_cols > max_candidate_cols) {
        payload.status_override = "too_large";
        payload.result.ok = false;
        payload.result.stats.no_path_count = 1;
        return payload;
    }
    if (payload.candidate_cols == 0) {
        const bool zero_syndrome = popcount_limbs(syndrome, model.n_limbs) == 0;
        payload.result.ok = zero_syndrome;
        payload.result.stats.no_path_count = zero_syndrome ? 0 : 1;
        if (zero_syndrome) {
            LogicalMask zero{};
            payload.result.logical_hat = zero;
            payload.result.log_evidence = 0.0;
            payload.result.terminal_log_masses[zero] = 0.0;
            payload.result.terminal_top_log_mass_gap = std::numeric_limits<double>::infinity();
        }
        return payload;
    }

    const double decode_started = now_seconds();
    payload.result = decode_native(
        reduced.model,
        syndrome,
        K,
        Delta,
        score_alpha,
        metric_mode,
        int_metric_scale
    );
    payload.decode_time_s = now_seconds() - decode_started;
    return payload;
}

void validate_decode_parameters(int K, double Delta, double score_alpha);

double choice_child_parity(
    const State& parent,
    const ChoiceColumn& column,
    const ChoiceOption& option,
    const std::array<std::uint64_t, kMaxLimbs>& candidate_det,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome
) {
    double value = parent.parity_logodds;
    for (const RowTerm& term : column.before_terms) {
        const std::size_t limb_index = static_cast<std::size_t>(term.limb);
        const std::uint64_t bit = (parent.key.det[limb_index] ^ syndrome[limb_index]) & term.bit;
        if (bit != 0) value -= term.parity;
    }
    for (const RowTerm& term : column.after_terms) {
        const std::size_t limb_index = static_cast<std::size_t>(term.limb);
        const std::uint64_t bit = (candidate_det[limb_index] ^ syndrome[limb_index]) & term.bit;
        if (bit != 0) value += term.parity;
    }
    (void)option;
    return value;
}

DecodeResult decode_choice_native(
    const ChoiceNativeModel& model,
    const std::array<std::uint64_t, kMaxLimbs>& syndrome,
    int K,
    double Delta,
    double score_alpha
) {
    validate_decode_parameters(K, Delta, score_alpha);
    const double total_started = now_seconds();
    DecodeResult result;
    std::vector<State> states;
    states.push_back(State{Key{}, 0.0, 0.0});
    std::vector<Candidate> merged;
    std::vector<std::size_t> survivor_indices;

    for (std::size_t column_index = 0; column_index < model.columns.size(); ++column_index) {
        const ChoiceColumn& column = model.columns[column_index];
        result.stats.processed_columns = static_cast<int>(column_index) + 1;
        std::map<Key, Candidate, KeyLess> merged_by_key{KeyLess{model.n_limbs}};
        const double transition_started = model.collect_phase_timing ? now_seconds() : 0.0;
        for (const State& parent : states) {
            for (const ChoiceOption& option : column.options) {
                if (!std::isfinite(option.log_prior)) continue;
                result.stats.transition_evals += 1ULL;
                std::array<std::uint64_t, kMaxLimbs> candidate_det{};
                bool accepted = true;
                for (int limb = 0; limb < model.n_limbs; ++limb) {
                    const std::size_t limb_index = static_cast<std::size_t>(limb);
                    candidate_det[limb_index] = parent.key.det[limb_index] ^ option.detector[limb_index];
                }
                for (int index = 0; index < column.close_nonzero_count; ++index) {
                    const int limb = column.close_nonzero_limbs[static_cast<std::size_t>(index)];
                    const std::size_t limb_index = static_cast<std::size_t>(limb);
                    if (((candidate_det[limb_index] ^ syndrome[limb_index]) & column.close_mask[limb_index]) != 0) {
                        accepted = false;
                        break;
                    }
                }
                if (!accepted) continue;

                Key child_key{};
                child_key.logical = logical_xor(parent.key.logical, option.logical, model.n_logical_limbs);
                for (int limb = 0; limb < model.n_limbs; ++limb) {
                    const std::size_t limb_index = static_cast<std::size_t>(limb);
                    child_key.det[limb_index] = candidate_det[limb_index] & column.active_mask[limb_index];
                }
                Candidate candidate;
                candidate.key = child_key;
                candidate.logmass = parent.logmass + option.log_prior;
                candidate.parity_logodds = choice_child_parity(parent, column, option, candidate_det, syndrome);
                auto inserted = merged_by_key.emplace(child_key, candidate);
                if (!inserted.second) {
                    inserted.first->second.logmass =
                        logaddexp_pair(inserted.first->second.logmass, candidate.logmass);
                }
            }
        }
        if (model.collect_phase_timing) {
            result.stats.transition_time_s += now_seconds() - transition_started;
        }

        const std::uint64_t candidate_count = static_cast<std::uint64_t>(merged_by_key.size());
        result.stats.max_pre_prune_state_count =
            std::max(result.stats.max_pre_prune_state_count, candidate_count);
        result.stats.sum_pre_prune_state_count += candidate_count;
        if (merged_by_key.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }

        const double prune_started = model.collect_phase_timing ? now_seconds() : 0.0;
        merged.clear();
        merged.reserve(merged_by_key.size());
        for (auto& item : merged_by_key) {
            merged.push_back(item.second);
        }
        double best_score = -std::numeric_limits<double>::infinity();
        for (Candidate& candidate : merged) {
            candidate.score = candidate.logmass + score_alpha * candidate.parity_logodds;
            if (candidate.score > best_score) best_score = candidate.score;
        }
        const double cutoff = best_score - Delta;
        survivor_indices.clear();
        survivor_indices.reserve(merged.size());
        for (std::size_t index = 0; index < merged.size(); ++index) {
            if (merged[index].score >= cutoff) survivor_indices.push_back(index);
        }

        CandidateBetter better{model.n_limbs};
        auto better_index = [&](std::size_t lhs, std::size_t rhs) {
            return better(merged[lhs], merged[rhs]);
        };
        if (static_cast<int>(survivor_indices.size()) > K) {
            std::nth_element(
                survivor_indices.begin(),
                survivor_indices.begin() + K,
                survivor_indices.end(),
                better_index
            );
            survivor_indices.resize(static_cast<std::size_t>(K));
        }
        std::sort(survivor_indices.begin(), survivor_indices.end(), better_index);

        states.clear();
        states.reserve(survivor_indices.size());
        for (std::size_t survivor_index : survivor_indices) {
            const Candidate& survivor = merged[survivor_index];
            states.push_back(State{survivor.key, survivor.logmass, survivor.parity_logodds});
        }
        if (model.collect_phase_timing) {
            result.stats.prune_time_s += now_seconds() - prune_started;
        }

        const std::uint64_t post_count = static_cast<std::uint64_t>(states.size());
        result.stats.max_post_prune_state_count =
            std::max(result.stats.max_post_prune_state_count, post_count);
        result.stats.sum_post_prune_state_count += post_count;
        if (states.empty()) {
            result.stats.no_path_count = 1;
            result.stats.total_time_s = now_seconds() - total_started;
            result.ok = false;
            return result;
        }
    }

    for (const State& state : states) {
        if (!det_zero(state.key, model.n_limbs)) continue;
        auto found = result.terminal_log_masses.find(state.key.logical);
        if (found == result.terminal_log_masses.end()) {
            result.terminal_log_masses[state.key.logical] = state.logmass;
        } else {
            found->second = logaddexp_pair(found->second, state.logmass);
        }
    }
    if (result.terminal_log_masses.empty()) {
        result.ok = false;
        result.stats.no_path_count = 1;
        result.stats.total_time_s = now_seconds() - total_started;
        return result;
    }

    result.ok = true;
    double top1 = -std::numeric_limits<double>::infinity();
    double top2 = -std::numeric_limits<double>::infinity();
    bool have_hat = false;
    result.log_evidence = -std::numeric_limits<double>::infinity();
    for (const auto& item : result.terminal_log_masses) {
        const LogicalMask& logical = item.first;
        const double logmass = item.second;
        result.log_evidence = logaddexp_pair(result.log_evidence, logmass);
        if (!have_hat || logmass > top1 || (logmass == top1 && logical < result.logical_hat)) {
            top2 = top1;
            top1 = logmass;
            result.logical_hat = logical;
            have_hat = true;
        } else if (logmass > top2) {
            top2 = logmass;
        }
    }
    result.terminal_top_log_mass_gap =
        std::isfinite(top2) ? top1 - top2 : std::numeric_limits<double>::infinity();
    result.stats.no_path_count = 0;
    result.stats.total_time_s = now_seconds() - total_started;
    return result;
}

double committee_terminal_gap_key(const DecodeResult& result) {
    if (std::isnan(result.terminal_top_log_mass_gap)) {
        return -std::numeric_limits<double>::infinity();
    }
    return result.terminal_top_log_mass_gap;
}

double committee_top1_posterior_key(const DecodeResult& result) {
    if (!result.ok || !std::isfinite(result.log_evidence) || result.terminal_log_masses.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    double top = -std::numeric_limits<double>::infinity();
    for (const auto& item : result.terminal_log_masses) {
        if (item.second > top) top = item.second;
    }
    const double posterior = std::exp(top - result.log_evidence);
    return std::isfinite(posterior) ? posterior : -std::numeric_limits<double>::infinity();
}

bool committee_prefers(
    const DecodeResult& candidate,
    bool candidate_forward,
    const DecodeResult& incumbent,
    bool incumbent_forward
) {
    const int candidate_status = candidate.ok ? 2 : 1;
    const int incumbent_status = incumbent.ok ? 2 : 1;
    if (candidate_status != incumbent_status) return candidate_status > incumbent_status;

    const double candidate_evidence =
        candidate.ok && std::isfinite(candidate.log_evidence)
            ? candidate.log_evidence
            : -std::numeric_limits<double>::infinity();
    const double incumbent_evidence =
        incumbent.ok && std::isfinite(incumbent.log_evidence)
            ? incumbent.log_evidence
            : -std::numeric_limits<double>::infinity();
    if (candidate_evidence != incumbent_evidence) return candidate_evidence > incumbent_evidence;

    const double candidate_gap = committee_terminal_gap_key(candidate);
    const double incumbent_gap = committee_terminal_gap_key(incumbent);
    if (candidate_gap != incumbent_gap) return candidate_gap > incumbent_gap;

    const double candidate_top1 = committee_top1_posterior_key(candidate);
    const double incumbent_top1 = committee_top1_posterior_key(incumbent);
    if (candidate_top1 != incumbent_top1) return candidate_top1 > incumbent_top1;

    if (candidate_forward != incumbent_forward) return candidate_forward;
    return false;
}

std::string first_stage_status(const FirstStageDecodePayload& payload) {
    if (!payload.status_override.empty()) {
        return payload.status_override;
    }
    return payload.result.ok ? std::string("ok") : std::string("no_path");
}

bool first_stage_payload_ok(const FirstStageDecodePayload& payload) {
    return first_stage_status(payload) == "ok" && payload.result.ok;
}

std::uint64_t first_stage_transition_evals_total(const Stage1NocapStage2Payload& payload) {
    return payload.stage1_forward.result.stats.transition_evals
        + payload.stage1_backward.result.stats.transition_evals;
}

std::uint64_t final_transition_evals_total(const Stage1NocapStage2Payload& payload) {
    const std::uint64_t stage1_total = first_stage_transition_evals_total(payload);
    if (!payload.used_stage2) {
        return stage1_total;
    }
    return stage1_total
        + payload.final_forward.stats.transition_evals
        + payload.final_backward.stats.transition_evals;
}

std::uint64_t stage2_transition_evals_total(const Stage1NocapStage2Payload& payload) {
    if (!payload.used_stage2) return 0;
    return payload.final_forward.stats.transition_evals + payload.final_backward.stats.transition_evals;
}

Stage1NocapStage2Payload decode_stage1_nocap_stage2_native(
    const NativeModel& forward_model,
    const NativeModel& backward_model,
    const std::array<std::uint64_t, kMaxLimbs>& forward_syndrome,
    const std::array<std::uint64_t, kMaxLimbs>& backward_syndrome,
    int K,
    double Delta,
    double score_alpha,
    MetricMode metric_mode,
    int int_metric_scale
) {
    validate_decode_parameters(K, Delta, score_alpha);
    validate_metric_options(metric_mode, int_metric_scale);

    Stage1NocapStage2Payload payload;
    payload.stage1_forward = decode_overlap1_first_stage_native(
        forward_model,
        forward_syndrome,
        K,
        Delta,
        score_alpha,
        metric_mode,
        int_metric_scale,
        0
    );
    payload.stage1_backward = decode_overlap1_first_stage_native(
        backward_model,
        backward_syndrome,
        K,
        Delta,
        score_alpha,
        metric_mode,
        int_metric_scale,
        0
    );

    const bool choose_stage1_backward = committee_prefers(
        payload.stage1_backward.result,
        false,
        payload.stage1_forward.result,
        true
    );
    payload.stage1_selected_forward = !choose_stage1_backward;
    const DecodeResult& stage1_selected =
        payload.stage1_selected_forward ? payload.stage1_forward.result : payload.stage1_backward.result;
    const bool forward_ok = first_stage_payload_ok(payload.stage1_forward);
    const bool backward_ok = first_stage_payload_ok(payload.stage1_backward);
    payload.stage1_agree =
        forward_ok &&
        backward_ok &&
        logical_equal_limited(
            payload.stage1_forward.result.logical_hat,
            payload.stage1_backward.result.logical_hat,
            forward_model.n_logical_limbs
        );

    if (payload.stage1_agree) {
        payload.used_stage2 = false;
        payload.stage1_status = "ok";
        payload.final_forward = payload.stage1_forward.result;
        payload.final_backward = payload.stage1_backward.result;
        payload.selected_forward = payload.stage1_selected_forward;
        payload.selected = stage1_selected;
        return payload;
    }

    payload.used_stage2 = true;
    payload.stage1_status = "direction_agreement_veto";
    payload.final_forward = decode_native(
        forward_model,
        forward_syndrome,
        K,
        Delta,
        score_alpha,
        metric_mode,
        int_metric_scale
    );
    payload.final_backward = decode_native(
        backward_model,
        backward_syndrome,
        K,
        Delta,
        score_alpha,
        metric_mode,
        int_metric_scale
    );
    const bool choose_stage2_backward = committee_prefers(payload.final_backward, false, payload.final_forward, true);
    payload.selected_forward = !choose_stage2_backward;
    payload.selected = payload.selected_forward ? payload.final_forward : payload.final_backward;
    return payload;
}

void validate_decode_parameters(int K, double Delta, double score_alpha) {
    if (K <= 0) {
        throw std::runtime_error("K must be positive");
    }
    if (Delta < 0.0) {
        throw std::runtime_error("Delta must be non-negative");
    }
    if (!std::isfinite(score_alpha) || score_alpha < 0.0) {
        throw std::runtime_error("score_alpha must be finite and non-negative");
    }
}

void record_first_exception(
    std::exception_ptr exc,
    std::exception_ptr& first_exception,
    std::mutex& exception_mutex
) {
    std::lock_guard<std::mutex> lock(exception_mutex);
    if (!first_exception) {
        first_exception = exc;
    }
}

void decode_many_native_with_optional_threads(
    const NativeModel& model,
    const std::vector<std::array<std::uint64_t, kMaxLimbs>>& syndromes,
    int K,
    double Delta,
    double score_alpha,
    MetricMode metric_mode,
    int int_metric_scale,
    std::vector<DecodeResult>& results
) {
    validate_decode_parameters(K, Delta, score_alpha);
    validate_metric_options(metric_mode, int_metric_scale);
    if (syndromes.size() != results.size()) {
        throw std::runtime_error("decode_many result buffer size mismatch");
    }
    const bool disable_workspace_reuse = batch_workspace_reuse_disabled();
    const std::size_t thread_count = native_batch_thread_count(syndromes.size());

    auto decode_range = [&](std::size_t begin, std::size_t end) {
        Compact4Workspace compact_workspace;
        FullWorkspace full_workspace;
        MaxLogIntWorkspace int_workspace;
        for (std::size_t index = begin; index < end; ++index) {
            if (disable_workspace_reuse) {
                results[index] = decode_native(
                    model,
                    syndromes[index],
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale
                );
            } else {
                results[index] = decode_native_with_workspace(
                    model,
                    syndromes[index],
                    K,
                    Delta,
                    score_alpha,
                    compact_workspace,
                    full_workspace,
                    int_workspace,
                    metric_mode,
                    int_metric_scale
                );
            }
        }
    };

    if (thread_count <= 1 || syndromes.size() <= 1) {
        decode_range(0, syndromes.size());
        return;
    }

    std::atomic<std::size_t> next_index{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    std::exception_ptr decode_error;
    std::mutex exception_mutex;
    for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        threads.emplace_back([&]() {
            try {
                Compact4Workspace compact_workspace;
                FullWorkspace full_workspace;
                MaxLogIntWorkspace int_workspace;
                while (true) {
                    const std::size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
                    if (index >= syndromes.size()) {
                        break;
                    }
                    if (disable_workspace_reuse) {
                        results[index] = decode_native(
                            model,
                            syndromes[index],
                            K,
                            Delta,
                            score_alpha,
                            metric_mode,
                            int_metric_scale
                        );
                    } else {
                        results[index] = decode_native_with_workspace(
                            model,
                            syndromes[index],
                            K,
                            Delta,
                            score_alpha,
                            compact_workspace,
                            full_workspace,
                            int_workspace,
                            metric_mode,
                            int_metric_scale
                        );
                    }
                }
            } catch (...) {
                record_first_exception(std::current_exception(), decode_error, exception_mutex);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (decode_error) {
        std::rethrow_exception(decode_error);
    }
}

void decode_many_choice_native_with_optional_threads(
    const ChoiceNativeModel& model,
    const std::vector<std::array<std::uint64_t, kMaxLimbs>>& syndromes,
    int K,
    double Delta,
    double score_alpha,
    std::vector<DecodeResult>& results
) {
    validate_decode_parameters(K, Delta, score_alpha);
    if (syndromes.size() != results.size()) {
        throw std::runtime_error("decode_many_choice result buffer size mismatch");
    }
    const std::size_t thread_count = native_batch_thread_count(syndromes.size());

    auto decode_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            results[index] = decode_choice_native(model, syndromes[index], K, Delta, score_alpha);
        }
    };

    if (thread_count <= 1 || syndromes.size() <= 1) {
        decode_range(0, syndromes.size());
        return;
    }

    std::atomic<std::size_t> next_index{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    std::exception_ptr decode_error;
    std::mutex exception_mutex;
    for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        threads.emplace_back([&]() {
            try {
                while (true) {
                    const std::size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
                    if (index >= syndromes.size()) {
                        break;
                    }
                    results[index] = decode_choice_native(model, syndromes[index], K, Delta, score_alpha);
                }
            } catch (...) {
                record_first_exception(std::current_exception(), decode_error, exception_mutex);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (decode_error) {
        std::rethrow_exception(decode_error);
    }
}

void decode_many_select_native_with_optional_threads(
    const NativeModel& forward_model,
    const NativeModel& backward_model,
    const std::vector<std::array<std::uint64_t, kMaxLimbs>>& forward_syndromes,
    const std::vector<std::array<std::uint64_t, kMaxLimbs>>& backward_syndromes,
    int K,
    double Delta,
    double score_alpha,
    MetricMode metric_mode,
    int int_metric_scale,
    std::vector<DecodeResult>& forward_results,
    std::vector<DecodeResult>& backward_results,
    std::vector<unsigned char>& selected_forward
) {
    validate_decode_parameters(K, Delta, score_alpha);
    validate_metric_options(metric_mode, int_metric_scale);
    if (forward_syndromes.size() != backward_syndromes.size()) {
        throw std::runtime_error("forward/backward syndrome batches must have the same length");
    }
    if (
        forward_syndromes.size() != forward_results.size() ||
        forward_syndromes.size() != backward_results.size() ||
        forward_syndromes.size() != selected_forward.size()
    ) {
        throw std::runtime_error("decode_many_select result buffer size mismatch");
    }

    const bool disable_workspace_reuse = batch_workspace_reuse_disabled();
    const std::size_t job_count = forward_syndromes.size() * 2U;
    const std::size_t thread_count = native_batch_thread_count(job_count);

    auto decode_range = [&](std::size_t begin, std::size_t end) {
        Compact4Workspace forward_compact_workspace;
        Compact4Workspace backward_compact_workspace;
        FullWorkspace forward_full_workspace;
        FullWorkspace backward_full_workspace;
        MaxLogIntWorkspace forward_int_workspace;
        MaxLogIntWorkspace backward_int_workspace;
        for (std::size_t index = begin; index < end; ++index) {
            DecodeResult forward;
            DecodeResult backward;
            if (disable_workspace_reuse) {
                forward = decode_native(
                    forward_model,
                    forward_syndromes[index],
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale
                );
                backward = decode_native(
                    backward_model,
                    backward_syndromes[index],
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale
                );
            } else {
                forward = decode_native_with_workspace(
                    forward_model,
                    forward_syndromes[index],
                    K,
                    Delta,
                    score_alpha,
                    forward_compact_workspace,
                    forward_full_workspace,
                    forward_int_workspace,
                    metric_mode,
                    int_metric_scale
                );
                backward = decode_native_with_workspace(
                    backward_model,
                    backward_syndromes[index],
                    K,
                    Delta,
                    score_alpha,
                    backward_compact_workspace,
                    backward_full_workspace,
                    backward_int_workspace,
                    metric_mode,
                    int_metric_scale
                );
            }
            const bool choose_backward = committee_prefers(backward, false, forward, true);
            selected_forward[index] = choose_backward ? 0U : 1U;
            forward_results[index] = std::move(forward);
            backward_results[index] = std::move(backward);
        }
    };

    if (thread_count <= 1 || forward_syndromes.size() <= 1) {
        decode_range(0, forward_syndromes.size());
        return;
    }

    std::atomic<std::size_t> next_job{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    std::exception_ptr decode_error;
    std::mutex exception_mutex;
    for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        threads.emplace_back([&]() {
            try {
                Compact4Workspace forward_compact_workspace;
                Compact4Workspace backward_compact_workspace;
                FullWorkspace forward_full_workspace;
                FullWorkspace backward_full_workspace;
                MaxLogIntWorkspace forward_int_workspace;
                MaxLogIntWorkspace backward_int_workspace;
                while (true) {
                    const std::size_t job = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (job >= job_count) {
                        break;
                    }
                    if (job < forward_syndromes.size()) {
                        const std::size_t index = job;
                        if (disable_workspace_reuse) {
                            forward_results[index] = decode_native(
                                forward_model,
                                forward_syndromes[index],
                                K,
                                Delta,
                                score_alpha,
                                metric_mode,
                                int_metric_scale
                            );
                        } else {
                            forward_results[index] = decode_native_with_workspace(
                                forward_model,
                                forward_syndromes[index],
                                K,
                                Delta,
                                score_alpha,
                                forward_compact_workspace,
                                forward_full_workspace,
                                forward_int_workspace,
                                metric_mode,
                                int_metric_scale
                            );
                        }
                    } else {
                        const std::size_t index = job - forward_syndromes.size();
                        if (disable_workspace_reuse) {
                            backward_results[index] = decode_native(
                                backward_model,
                                backward_syndromes[index],
                                K,
                                Delta,
                                score_alpha,
                                metric_mode,
                                int_metric_scale
                            );
                        } else {
                            backward_results[index] = decode_native_with_workspace(
                                backward_model,
                                backward_syndromes[index],
                                K,
                                Delta,
                                score_alpha,
                                backward_compact_workspace,
                                backward_full_workspace,
                                backward_int_workspace,
                                metric_mode,
                                int_metric_scale
                            );
                        }
                    }
                }
            } catch (...) {
                record_first_exception(std::current_exception(), decode_error, exception_mutex);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (decode_error) {
        std::rethrow_exception(decode_error);
    }
    for (std::size_t index = 0; index < forward_syndromes.size(); ++index) {
        const bool choose_backward = committee_prefers(backward_results[index], false, forward_results[index], true);
        selected_forward[index] = choose_backward ? 0U : 1U;
    }
}

void decode_many_stage1_nocap_stage2_native_with_optional_threads(
    const NativeModel& forward_model,
    const NativeModel& backward_model,
    const std::vector<std::array<std::uint64_t, kMaxLimbs>>& forward_syndromes,
    const std::vector<std::array<std::uint64_t, kMaxLimbs>>& backward_syndromes,
    int K,
    double Delta,
    double score_alpha,
    MetricMode metric_mode,
    int int_metric_scale,
    std::vector<Stage1NocapStage2Payload>& results
) {
    validate_decode_parameters(K, Delta, score_alpha);
    validate_metric_options(metric_mode, int_metric_scale);
    if (forward_syndromes.size() != backward_syndromes.size()) {
        throw std::runtime_error("forward/backward syndrome batches must have the same length");
    }
    if (forward_syndromes.size() != results.size()) {
        throw std::runtime_error("stage1_nocap_stage2 result buffer size mismatch");
    }

    const std::size_t job_count = forward_syndromes.size();
    const std::size_t thread_count = native_batch_thread_count(job_count);
    auto decode_index = [&](std::size_t index) {
        results[index] = decode_stage1_nocap_stage2_native(
            forward_model,
            backward_model,
            forward_syndromes[index],
            backward_syndromes[index],
            K,
            Delta,
            score_alpha,
            metric_mode,
            int_metric_scale
        );
    };

    if (thread_count <= 1 || job_count <= 1) {
        for (std::size_t index = 0; index < job_count; ++index) {
            decode_index(index);
        }
        return;
    }

    std::atomic<std::size_t> next_job{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    std::exception_ptr decode_error;
    std::mutex exception_mutex;
    for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
        threads.emplace_back([&]() {
            try {
                while (true) {
                    const std::size_t job = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (job >= job_count) {
                        break;
                    }
                    decode_index(job);
                }
            } catch (...) {
                record_first_exception(std::current_exception(), decode_error, exception_mutex);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (decode_error) {
        std::rethrow_exception(decode_error);
    }
}

PyObject* stats_to_dict(const DecodeStats& stats) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) return nullptr;
    if (
        !dict_set_steal(dict, "processed_columns", PyLong_FromLong(stats.processed_columns)) ||
        !dict_set_steal(dict, "transition_evals", PyLong_FromUnsignedLongLong(stats.transition_evals)) ||
        !dict_set_steal(dict, "max_pre_prune_state_count", PyLong_FromUnsignedLongLong(stats.max_pre_prune_state_count)) ||
        !dict_set_steal(dict, "max_post_prune_state_count", PyLong_FromUnsignedLongLong(stats.max_post_prune_state_count)) ||
        !dict_set_steal(dict, "sum_pre_prune_state_count", PyLong_FromUnsignedLongLong(stats.sum_pre_prune_state_count)) ||
        !dict_set_steal(dict, "sum_post_prune_state_count", PyLong_FromUnsignedLongLong(stats.sum_post_prune_state_count)) ||
        !dict_set_steal(dict, "no_path_count", PyLong_FromLong(stats.no_path_count)) ||
        !dict_set_steal(dict, "transition_time_s", PyFloat_FromDouble(stats.transition_time_s)) ||
        !dict_set_steal(dict, "merge_time_s", PyFloat_FromDouble(stats.merge_time_s)) ||
        !dict_set_steal(dict, "prune_time_s", PyFloat_FromDouble(stats.prune_time_s)) ||
        !dict_set_steal(dict, "total_time_s", PyFloat_FromDouble(stats.total_time_s)) ||
        !dict_set_steal(
            dict,
            "profile_no_merge_transition_columns",
            PyLong_FromUnsignedLongLong(stats.profile_no_merge_transition_columns)
        ) ||
        !dict_set_steal(
            dict,
            "profile_split_merge_columns",
            PyLong_FromUnsignedLongLong(stats.profile_split_merge_columns)
        ) ||
        !dict_set_steal(
            dict,
            "profile_generic_merge_columns",
            PyLong_FromUnsignedLongLong(stats.profile_generic_merge_columns)
        ) ||
        !dict_set_steal(
            dict,
            "profile_emit_child_calls",
            PyLong_FromUnsignedLongLong(stats.profile_emit_child_calls)
        ) ||
        !dict_set_steal(
            dict,
            "profile_merge_duplicate_count",
            PyLong_FromUnsignedLongLong(stats.profile_merge_duplicate_count)
        ) ||
        !dict_set_steal(
            dict,
            "profile_hash_probe_total",
            PyLong_FromUnsignedLongLong(stats.profile_hash_probe_total)
        ) ||
        !dict_set_steal(
            dict,
            "profile_hash_probe_max",
            PyLong_FromUnsignedLongLong(stats.profile_hash_probe_max)
        ) ||
        !dict_set_steal(
            dict,
            "profile_score_evals",
            PyLong_FromUnsignedLongLong(stats.profile_score_evals)
        ) ||
        !dict_set_steal(
            dict,
            "profile_nth_element_calls",
            PyLong_FromUnsignedLongLong(stats.profile_nth_element_calls)
        )
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* compact_stats_to_dict(const DecodeStats& stats) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) return nullptr;
    if (
        !dict_set_steal(dict, "processed_columns", PyLong_FromLong(stats.processed_columns)) ||
        !dict_set_steal(dict, "transition_evals", PyLong_FromUnsignedLongLong(stats.transition_evals)) ||
        !dict_set_steal(dict, "max_pre_prune_state_count", PyLong_FromUnsignedLongLong(stats.max_pre_prune_state_count)) ||
        !dict_set_steal(dict, "max_post_prune_state_count", PyLong_FromUnsignedLongLong(stats.max_post_prune_state_count)) ||
        !dict_set_steal(dict, "sum_pre_prune_state_count", PyLong_FromUnsignedLongLong(stats.sum_pre_prune_state_count)) ||
        !dict_set_steal(dict, "sum_post_prune_state_count", PyLong_FromUnsignedLongLong(stats.sum_post_prune_state_count)) ||
        !dict_set_steal(dict, "no_path_count", PyLong_FromLong(stats.no_path_count))
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* result_to_dict(const DecodeResult& result, int n_logical_limbs) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) return nullptr;
    if (!dict_set_steal(dict, "status", PyUnicode_FromString(result.ok ? "ok" : "no_path"))) {
        Py_DECREF(dict);
        return nullptr;
    }
    if (result.ok) {
        if (!dict_set_steal(dict, "logical_hat", logical_to_pylong(result.logical_hat, n_logical_limbs))) {
            Py_DECREF(dict);
            return nullptr;
        }
    } else {
        PyObject* none_value = Py_None;
        Py_INCREF(none_value);
        if (!dict_set_steal(dict, "logical_hat", none_value)) {
            Py_DECREF(dict);
            return nullptr;
        }
    }
    if (
        !dict_set_steal(dict, "log_evidence", PyFloat_FromDouble(result.log_evidence)) ||
        !dict_set_steal(dict, "terminal_top_log_mass_gap", PyFloat_FromDouble(result.terminal_top_log_mass_gap))
    ) {
        Py_DECREF(dict);
        return nullptr;
    }

    PyObject* masses = PyDict_New();
    if (masses == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }
    for (const auto& item : result.terminal_log_masses) {
        PyObject* key = logical_to_pylong(item.first, n_logical_limbs);
        PyObject* value = PyFloat_FromDouble(item.second);
        if (key == nullptr || value == nullptr || PyDict_SetItem(masses, key, value) != 0) {
            Py_XDECREF(key);
            Py_XDECREF(value);
            Py_DECREF(masses);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(key);
        Py_DECREF(value);
    }
    if (PyDict_SetItemString(dict, "terminal_log_masses", masses) != 0) {
        Py_DECREF(masses);
        Py_DECREF(dict);
        return nullptr;
    }
    Py_DECREF(masses);

    PyObject* stats = stats_to_dict(result.stats);
    if (stats == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }
    if (PyDict_SetItemString(dict, "stats", stats) != 0) {
        Py_DECREF(stats);
        Py_DECREF(dict);
        return nullptr;
    }
    Py_DECREF(stats);
    return dict;
}

PyObject* first_stage_payload_to_dict(const FirstStageDecodePayload& payload, int n_logical_limbs) {
    PyObject* dict = result_to_dict(payload.result, n_logical_limbs);
    if (dict == nullptr) return nullptr;
    if (!payload.status_override.empty()) {
        if (!dict_set_steal(dict, "status", PyUnicode_FromString(payload.status_override.c_str()))) {
            Py_DECREF(dict);
            return nullptr;
        }
    }
    if (
        !dict_set_steal(dict, "candidate_cols", PyLong_FromLong(payload.candidate_cols)) ||
        !dict_set_steal(dict, "reduced_rows", PyLong_FromLong(payload.reduced_rows)) ||
        !dict_set_steal(dict, "active_syndrome_weight", PyLong_FromLong(payload.active_syndrome_weight)) ||
        !dict_set_steal(dict, "uncovered_active_rows", PyLong_FromLong(payload.uncovered_active_rows)) ||
        !dict_set_steal(dict, "build_time_s", PyFloat_FromDouble(payload.build_time_s)) ||
        !dict_set_steal(dict, "decode_time_s", PyFloat_FromDouble(payload.decode_time_s))
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* result_to_compact_dict(const DecodeResult& result, int n_logical_limbs) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) return nullptr;
    if (!dict_set_steal(dict, "status", PyUnicode_FromString(result.ok ? "ok" : "no_path"))) {
        Py_DECREF(dict);
        return nullptr;
    }
    if (result.ok) {
        if (!dict_set_steal(dict, "logical_hat", logical_to_pylong(result.logical_hat, n_logical_limbs))) {
            Py_DECREF(dict);
            return nullptr;
        }
    } else {
        PyObject* none_value = Py_None;
        Py_INCREF(none_value);
        if (!dict_set_steal(dict, "logical_hat", none_value)) {
            Py_DECREF(dict);
            return nullptr;
        }
    }
    if (
        !dict_set_steal(dict, "log_evidence", PyFloat_FromDouble(result.log_evidence)) ||
        !dict_set_steal(dict, "terminal_top_log_mass_gap", PyFloat_FromDouble(result.terminal_top_log_mass_gap))
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    PyObject* stats = compact_stats_to_dict(result.stats);
    if (stats == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }
    if (PyDict_SetItemString(dict, "stats", stats) != 0) {
        Py_DECREF(stats);
        Py_DECREF(dict);
        return nullptr;
    }
    Py_DECREF(stats);
    return dict;
}

PyObject* selected_result_to_dict(
    const DecodeResult& selected,
    bool selected_forward,
    const DecodeResult& forward,
    const DecodeResult& backward,
    int n_logical_limbs
) {
    PyObject* dict = result_to_dict(selected, n_logical_limbs);
    if (dict == nullptr) return nullptr;
    const std::uint64_t total_transition_evals =
        forward.stats.transition_evals + backward.stats.transition_evals;
    if (
        !dict_set_steal(
            dict,
            "selected_direction",
            PyUnicode_FromString(selected_forward ? "forward" : "backward")
        ) ||
        !dict_set_steal(dict, "forward_status", PyUnicode_FromString(forward.ok ? "ok" : "no_path")) ||
        !dict_set_steal(dict, "backward_status", PyUnicode_FromString(backward.ok ? "ok" : "no_path")) ||
        !dict_set_steal(
            dict,
            "transition_evals_total",
            PyLong_FromUnsignedLongLong(total_transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "forward_transition_evals",
            PyLong_FromUnsignedLongLong(forward.stats.transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "backward_transition_evals",
            PyLong_FromUnsignedLongLong(backward.stats.transition_evals)
        )
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* selected_result_to_compact_dict(
    const DecodeResult& selected,
    bool selected_forward,
    const DecodeResult& forward,
    const DecodeResult& backward,
    int n_logical_limbs
) {
    PyObject* dict = result_to_compact_dict(selected, n_logical_limbs);
    if (dict == nullptr) return nullptr;
    const std::uint64_t total_transition_evals =
        forward.stats.transition_evals + backward.stats.transition_evals;
    if (
        !dict_set_steal(
            dict,
            "selected_direction",
            PyUnicode_FromString(selected_forward ? "forward" : "backward")
        ) ||
        !dict_set_steal(dict, "forward_status", PyUnicode_FromString(forward.ok ? "ok" : "no_path")) ||
        !dict_set_steal(dict, "backward_status", PyUnicode_FromString(backward.ok ? "ok" : "no_path")) ||
        !dict_set_steal(
            dict,
            "transition_evals_total",
            PyLong_FromUnsignedLongLong(total_transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "forward_transition_evals",
            PyLong_FromUnsignedLongLong(forward.stats.transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "backward_transition_evals",
            PyLong_FromUnsignedLongLong(backward.stats.transition_evals)
        )
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* selected_result_to_replay_dict(
    const DecodeResult& selected,
    bool selected_forward,
    const DecodeResult& forward,
    const DecodeResult& backward,
    int n_logical_limbs
) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) return nullptr;
    if (!dict_set_steal(dict, "status", PyUnicode_FromString(selected.ok ? "ok" : "no_path"))) {
        Py_DECREF(dict);
        return nullptr;
    }
    if (selected.ok) {
        if (!dict_set_steal(dict, "logical_hat", logical_to_pylong(selected.logical_hat, n_logical_limbs))) {
            Py_DECREF(dict);
            return nullptr;
        }
    } else {
        PyObject* none_value = Py_None;
        Py_INCREF(none_value);
        if (!dict_set_steal(dict, "logical_hat", none_value)) {
            Py_DECREF(dict);
            return nullptr;
        }
    }
    auto set_logical_hat_field = [&](const char* name, const DecodeResult& result) -> bool {
        if (result.ok) {
            return dict_set_steal(dict, name, logical_to_pylong(result.logical_hat, n_logical_limbs));
        }
        PyObject* none_value = Py_None;
        Py_INCREF(none_value);
        return dict_set_steal(dict, name, none_value);
    };

    const std::uint64_t total_transition_evals =
        forward.stats.transition_evals + backward.stats.transition_evals;
    if (
        !dict_set_steal(dict, "log_evidence", PyFloat_FromDouble(selected.log_evidence)) ||
        !dict_set_steal(dict, "terminal_top_log_mass_gap", PyFloat_FromDouble(selected.terminal_top_log_mass_gap)) ||
        !dict_set_steal(
            dict,
            "selected_direction",
            PyUnicode_FromString(selected_forward ? "forward" : "backward")
        ) ||
        !dict_set_steal(dict, "forward_status", PyUnicode_FromString(forward.ok ? "ok" : "no_path")) ||
        !dict_set_steal(dict, "backward_status", PyUnicode_FromString(backward.ok ? "ok" : "no_path")) ||
        !set_logical_hat_field("forward_logical_hat", forward) ||
        !set_logical_hat_field("backward_logical_hat", backward) ||
        !dict_set_steal(dict, "forward_log_evidence", PyFloat_FromDouble(forward.log_evidence)) ||
        !dict_set_steal(dict, "backward_log_evidence", PyFloat_FromDouble(backward.log_evidence)) ||
        !dict_set_steal(
            dict,
            "forward_terminal_top_log_mass_gap",
            PyFloat_FromDouble(forward.terminal_top_log_mass_gap)
        ) ||
        !dict_set_steal(
            dict,
            "backward_terminal_top_log_mass_gap",
            PyFloat_FromDouble(backward.terminal_top_log_mass_gap)
        ) ||
        !dict_set_steal(
            dict,
            "transition_evals_total",
            PyLong_FromUnsignedLongLong(total_transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "forward_transition_evals",
            PyLong_FromUnsignedLongLong(forward.stats.transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "backward_transition_evals",
            PyLong_FromUnsignedLongLong(backward.stats.transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "forward_max_post_prune_state_count",
            PyLong_FromUnsignedLongLong(forward.stats.max_post_prune_state_count)
        ) ||
        !dict_set_steal(
            dict,
            "backward_max_post_prune_state_count",
            PyLong_FromUnsignedLongLong(backward.stats.max_post_prune_state_count)
        ) ||
        !dict_set_steal(
            dict,
            "selected_transition_evals",
            PyLong_FromUnsignedLongLong(selected.stats.transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            "max_pre_prune_state_count",
            PyLong_FromUnsignedLongLong(selected.stats.max_pre_prune_state_count)
        ) ||
        !dict_set_steal(
            dict,
            "max_post_prune_state_count",
            PyLong_FromUnsignedLongLong(selected.stats.max_post_prune_state_count)
        ) ||
        !dict_set_steal(
            dict,
            "sum_pre_prune_state_count",
            PyLong_FromUnsignedLongLong(selected.stats.sum_pre_prune_state_count)
        ) ||
        !dict_set_steal(
            dict,
            "sum_post_prune_state_count",
            PyLong_FromUnsignedLongLong(selected.stats.sum_post_prune_state_count)
        ) ||
        !dict_set_steal(dict, "processed_columns", PyLong_FromLong(selected.stats.processed_columns)) ||
        !dict_set_steal(dict, "no_path_count", PyLong_FromLong(selected.stats.no_path_count))
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

bool dict_set_logical_or_none(PyObject* dict, const char* name, const DecodeResult& result, int n_logical_limbs) {
    if (result.ok) {
        return dict_set_steal(dict, name, logical_to_pylong(result.logical_hat, n_logical_limbs));
    }
    PyObject* none_value = Py_None;
    Py_INCREF(none_value);
    return dict_set_steal(dict, name, none_value);
}

bool dict_set_first_stage_payload(
    PyObject* dict,
    const char* prefix,
    const FirstStageDecodePayload& payload,
    int n_logical_limbs
) {
    const std::string status = first_stage_status(payload);
    const std::string status_key = std::string(prefix) + "_status";
    const std::string logical_key = std::string(prefix) + "_logical_hat";
    const std::string log_evidence_key = std::string(prefix) + "_log_evidence";
    const std::string gap_key = std::string(prefix) + "_terminal_top_log_mass_gap";
    const std::string transition_key = std::string(prefix) + "_transition_evals";
    const std::string max_post_key = std::string(prefix) + "_max_post_prune_state_count";
    const std::string candidate_key = std::string(prefix) + "_candidate_cols";
    const std::string reduced_rows_key = std::string(prefix) + "_reduced_rows";
    const std::string uncovered_key = std::string(prefix) + "_uncovered_active_rows";
    if (
        !dict_set_steal(dict, status_key.c_str(), PyUnicode_FromString(status.c_str())) ||
        !dict_set_logical_or_none(dict, logical_key.c_str(), payload.result, n_logical_limbs) ||
        !dict_set_steal(dict, log_evidence_key.c_str(), PyFloat_FromDouble(payload.result.log_evidence)) ||
        !dict_set_steal(dict, gap_key.c_str(), PyFloat_FromDouble(payload.result.terminal_top_log_mass_gap)) ||
        !dict_set_steal(
            dict,
            transition_key.c_str(),
            PyLong_FromUnsignedLongLong(payload.result.stats.transition_evals)
        ) ||
        !dict_set_steal(
            dict,
            max_post_key.c_str(),
            PyLong_FromUnsignedLongLong(payload.result.stats.max_post_prune_state_count)
        ) ||
        !dict_set_steal(dict, candidate_key.c_str(), PyLong_FromLong(payload.candidate_cols)) ||
        !dict_set_steal(dict, reduced_rows_key.c_str(), PyLong_FromLong(payload.reduced_rows)) ||
        !dict_set_steal(dict, uncovered_key.c_str(), PyLong_FromLong(payload.uncovered_active_rows))
    ) {
        return false;
    }
    return true;
}

PyObject* stage1_nocap_stage2_payload_to_replay_dict(
    const Stage1NocapStage2Payload& payload,
    int n_logical_limbs
) {
    PyObject* dict = selected_result_to_replay_dict(
        payload.selected,
        payload.selected_forward,
        payload.final_forward,
        payload.final_backward,
        n_logical_limbs
    );
    if (dict == nullptr) return nullptr;

    const DecodeResult& stage1_selected =
        payload.stage1_selected_forward ? payload.stage1_forward.result : payload.stage1_backward.result;
    const std::uint64_t stage1_total = first_stage_transition_evals_total(payload);
    const std::uint64_t stage2_total = stage2_transition_evals_total(payload);
    const std::uint64_t full_total = final_transition_evals_total(payload);
    const std::string stage1_selected_direction = payload.stage1_selected_forward ? "forward" : "backward";
    const std::string primary_status = payload.stage1_agree ? std::string("ok") : payload.stage1_status;
    if (
        !dict_set_steal(dict, "transition_evals_total", PyLong_FromUnsignedLongLong(full_total)) ||
        !dict_set_steal(dict, "primary_transition_evals_total", PyLong_FromUnsignedLongLong(stage1_total)) ||
        !dict_set_steal(dict, "escalation_transition_evals_total", PyLong_FromUnsignedLongLong(stage2_total)) ||
        !dict_set_steal(dict, "escalated", PyBool_FromLong(payload.used_stage2 ? 1 : 0)) ||
        !dict_set_steal(
            dict,
            "escalation_reason",
            PyUnicode_FromString(payload.used_stage2 ? payload.stage1_status.c_str() : "")
        ) ||
        !dict_set_steal(dict, "committee_disagreed", PyBool_FromLong(payload.stage1_agree ? 0 : 1)) ||
        !dict_set_steal(dict, "stage1_status", PyUnicode_FromString(payload.stage1_status.c_str())) ||
        !dict_set_steal(dict, "stage1_accept", PyBool_FromLong(payload.stage1_agree ? 1 : 0)) ||
        !dict_set_steal(
            dict,
            "stage1_selected_direction",
            PyUnicode_FromString(stage1_selected_direction.c_str())
        ) ||
        !dict_set_logical_or_none(dict, "stage1_logical_hat", stage1_selected, n_logical_limbs) ||
        !dict_set_steal(dict, "stage1_transition_evals_total", PyLong_FromUnsignedLongLong(stage1_total)) ||
        !dict_set_steal(dict, "stage2_transition_evals_total", PyLong_FromUnsignedLongLong(stage2_total)) ||
        !dict_set_steal(dict, "primary_status", PyUnicode_FromString(primary_status.c_str())) ||
        !dict_set_logical_or_none(dict, "primary_logical_hat", stage1_selected, n_logical_limbs) ||
        !dict_set_steal(
            dict,
            "primary_selected_direction",
            PyUnicode_FromString(stage1_selected_direction.c_str())
        ) ||
        !dict_set_first_stage_payload(dict, "stage1_forward", payload.stage1_forward, n_logical_limbs) ||
        !dict_set_first_stage_payload(dict, "stage1_backward", payload.stage1_backward, n_logical_limbs)
    ) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

enum class SelectPayloadKind {
    Full,
    Compact,
    Replay,
};

PyObject* py_make_model(PyObject*, PyObject* args) {
    PyObject* spec = nullptr;
    if (!PyArg_ParseTuple(args, "O", &spec)) {
        return nullptr;
    }
    try {
        std::unique_ptr<NativeModel> model = parse_model(spec);
        return PyCapsule_New(model.release(), "frontier_native_model", capsule_destructor);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_make_choice_model(PyObject*, PyObject* args) {
    PyObject* spec = nullptr;
    if (!PyArg_ParseTuple(args, "O", &spec)) {
        return nullptr;
    }
    try {
        std::unique_ptr<ChoiceNativeModel> model = parse_choice_model(spec);
        return PyCapsule_New(model.release(), "frontier_native_choice_model", choice_capsule_destructor);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* syndrome_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    const char* metric_mode_raw = "logsumexp_float";
    int int_metric_scale = 1024;
    if (!PyArg_ParseTuple(
            args,
            "OOidd|si",
            &capsule,
            &syndrome_obj,
            &K,
            &Delta,
            &score_alpha,
            &metric_mode_raw,
            &int_metric_scale
        )) {
        return nullptr;
    }
    try {
        const MetricMode metric_mode = parse_metric_mode(metric_mode_raw);
        NativeModel* model = model_from_capsule(capsule);
        const auto syndrome = parse_syndrome_limbs(syndrome_obj, model->n_limbs);
        DecodeResult result;
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                result = decode_native(*model, syndrome, K, Delta, score_alpha, metric_mode, int_metric_scale);
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        return result_to_dict(result, model->n_logical_limbs);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode_overlap1_first_stage(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* syndrome_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    const char* metric_mode_raw = "logsumexp_float";
    int int_metric_scale = 1024;
    int max_candidate_cols = 0;
    if (!PyArg_ParseTuple(
            args,
            "OOidd|sii",
            &capsule,
            &syndrome_obj,
            &K,
            &Delta,
            &score_alpha,
            &metric_mode_raw,
            &int_metric_scale,
            &max_candidate_cols
        )) {
        return nullptr;
    }
    try {
        const MetricMode metric_mode = parse_metric_mode(metric_mode_raw);
        NativeModel* model = model_from_capsule(capsule);
        const auto syndrome = parse_syndrome_limbs(syndrome_obj, model->n_limbs);
        FirstStageDecodePayload payload;
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                payload = decode_overlap1_first_stage_native(
                    *model,
                    syndrome,
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale,
                    max_candidate_cols
                );
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        return first_stage_payload_to_dict(payload, model->n_logical_limbs);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode_choice(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* syndrome_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    if (!PyArg_ParseTuple(args, "OOidd", &capsule, &syndrome_obj, &K, &Delta, &score_alpha)) {
        return nullptr;
    }
    try {
        ChoiceNativeModel* model = choice_model_from_capsule(capsule);
        const auto syndrome = parse_syndrome_limbs(syndrome_obj, model->n_limbs);
        DecodeResult result;
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                result = decode_choice_native(*model, syndrome, K, Delta, score_alpha);
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        return result_to_dict(result, model->n_logical_limbs);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode_many(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* syndromes_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    const char* metric_mode_raw = "logsumexp_float";
    int int_metric_scale = 1024;
    if (!PyArg_ParseTuple(
            args,
            "OOidd|si",
            &capsule,
            &syndromes_obj,
            &K,
            &Delta,
            &score_alpha,
            &metric_mode_raw,
            &int_metric_scale
        )) {
        return nullptr;
    }
    try {
        const MetricMode metric_mode = parse_metric_mode(metric_mode_raw);
        NativeModel* model = model_from_capsule(capsule);
        const auto syndromes = parse_many_syndrome_limbs(syndromes_obj, model->n_limbs);
        std::vector<DecodeResult> results(syndromes.size());
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                decode_many_native_with_optional_threads(
                    *model,
                    syndromes,
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale,
                    results
                );
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(results.size()));
        if (list == nullptr) return nullptr;
        for (std::size_t index = 0; index < results.size(); ++index) {
            PyObject* item = result_to_dict(results[index], model->n_logical_limbs);
            if (item == nullptr) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(index), item);
        }
        return list;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode_many_choice(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* syndromes_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    if (!PyArg_ParseTuple(args, "OOidd", &capsule, &syndromes_obj, &K, &Delta, &score_alpha)) {
        return nullptr;
    }
    try {
        ChoiceNativeModel* model = choice_model_from_capsule(capsule);
        const auto syndromes = parse_many_syndrome_limbs(syndromes_obj, model->n_limbs);
        std::vector<DecodeResult> results(syndromes.size());
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                decode_many_choice_native_with_optional_threads(*model, syndromes, K, Delta, score_alpha, results);
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(results.size()));
        if (list == nullptr) return nullptr;
        for (std::size_t index = 0; index < results.size(); ++index) {
            PyObject* item = result_to_dict(results[index], model->n_logical_limbs);
            if (item == nullptr) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(index), item);
        }
        return list;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode_many_select_impl(PyObject* args, SelectPayloadKind payload_kind) {
    PyObject* forward_capsule = nullptr;
    PyObject* backward_capsule = nullptr;
    PyObject* forward_syndromes_obj = nullptr;
    PyObject* backward_syndromes_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    const char* metric_mode_raw = "logsumexp_float";
    int int_metric_scale = 1024;
    if (!PyArg_ParseTuple(
            args,
            "OOOOidd|si",
            &forward_capsule,
            &backward_capsule,
            &forward_syndromes_obj,
            &backward_syndromes_obj,
            &K,
            &Delta,
            &score_alpha,
            &metric_mode_raw,
            &int_metric_scale
        )) {
        return nullptr;
    }
    try {
        const MetricMode metric_mode = parse_metric_mode(metric_mode_raw);
        NativeModel* forward_model = model_from_capsule(forward_capsule);
        NativeModel* backward_model = model_from_capsule(backward_capsule);
        const auto forward_syndromes = parse_many_syndrome_limbs(forward_syndromes_obj, forward_model->n_limbs);
        const auto backward_syndromes = parse_many_syndrome_limbs(backward_syndromes_obj, backward_model->n_limbs);
        if (forward_syndromes.size() != backward_syndromes.size()) {
            throw std::runtime_error("forward/backward syndrome batches must have the same length");
        }
        if (K <= 0) {
            throw std::runtime_error("K must be positive");
        }
        if (Delta < 0.0) {
            throw std::runtime_error("Delta must be non-negative");
        }
        if (!std::isfinite(score_alpha) || score_alpha < 0.0) {
            throw std::runtime_error("score_alpha must be finite and non-negative");
        }
        validate_metric_options(metric_mode, int_metric_scale);

        std::vector<DecodeResult> forward_results(forward_syndromes.size());
        std::vector<DecodeResult> backward_results(forward_syndromes.size());
        std::vector<unsigned char> selected_forward(forward_syndromes.size(), 1U);
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                decode_many_select_native_with_optional_threads(
                    *forward_model,
                    *backward_model,
                    forward_syndromes,
                    backward_syndromes,
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale,
                    forward_results,
                    backward_results,
                    selected_forward
                );
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(forward_results.size()));
        if (list == nullptr) return nullptr;
        for (std::size_t index = 0; index < forward_results.size(); ++index) {
            const DecodeResult& selected =
                selected_forward[index] != 0U ? forward_results[index] : backward_results[index];
            PyObject* item = nullptr;
            if (payload_kind == SelectPayloadKind::Replay) {
                item = selected_result_to_replay_dict(
                    selected,
                    selected_forward[index] != 0U,
                    forward_results[index],
                    backward_results[index],
                    forward_model->n_logical_limbs
                );
            } else if (payload_kind == SelectPayloadKind::Compact) {
                item = selected_result_to_compact_dict(
                    selected,
                    selected_forward[index] != 0U,
                    forward_results[index],
                    backward_results[index],
                    forward_model->n_logical_limbs
                );
            } else {
                item = selected_result_to_dict(
                    selected,
                    selected_forward[index] != 0U,
                    forward_results[index],
                    backward_results[index],
                    forward_model->n_logical_limbs
                );
            }
            if (item == nullptr) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(index), item);
        }
        return list;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_decode_many_select(PyObject*, PyObject* args) {
    return py_decode_many_select_impl(args, SelectPayloadKind::Full);
}

PyObject* py_decode_many_select_compact(PyObject*, PyObject* args) {
    return py_decode_many_select_impl(args, SelectPayloadKind::Compact);
}

PyObject* py_decode_many_select_replay(PyObject*, PyObject* args) {
    return py_decode_many_select_impl(args, SelectPayloadKind::Replay);
}

PyObject* py_decode_many_stage1_nocap_stage2_replay(PyObject*, PyObject* args) {
    PyObject* forward_capsule = nullptr;
    PyObject* backward_capsule = nullptr;
    PyObject* forward_syndromes_obj = nullptr;
    PyObject* backward_syndromes_obj = nullptr;
    int K = 0;
    double Delta = 0.0;
    double score_alpha = 0.8;
    const char* metric_mode_raw = "logsumexp_float";
    int int_metric_scale = 1024;
    if (!PyArg_ParseTuple(
            args,
            "OOOOidd|si",
            &forward_capsule,
            &backward_capsule,
            &forward_syndromes_obj,
            &backward_syndromes_obj,
            &K,
            &Delta,
            &score_alpha,
            &metric_mode_raw,
            &int_metric_scale
        )) {
        return nullptr;
    }
    try {
        const MetricMode metric_mode = parse_metric_mode(metric_mode_raw);
        NativeModel* forward_model = model_from_capsule(forward_capsule);
        NativeModel* backward_model = model_from_capsule(backward_capsule);
        const auto forward_syndromes = parse_many_syndrome_limbs(forward_syndromes_obj, forward_model->n_limbs);
        const auto backward_syndromes = parse_many_syndrome_limbs(backward_syndromes_obj, backward_model->n_limbs);
        if (forward_syndromes.size() != backward_syndromes.size()) {
            throw std::runtime_error("forward/backward syndrome batches must have the same length");
        }
        if (K <= 0) {
            throw std::runtime_error("K must be positive");
        }
        if (Delta < 0.0) {
            throw std::runtime_error("Delta must be non-negative");
        }
        if (!std::isfinite(score_alpha) || score_alpha < 0.0) {
            throw std::runtime_error("score_alpha must be finite and non-negative");
        }
        validate_metric_options(metric_mode, int_metric_scale);

        std::vector<Stage1NocapStage2Payload> results(forward_syndromes.size());
        std::exception_ptr decode_error;
        {
            Py_BEGIN_ALLOW_THREADS
            try {
                decode_many_stage1_nocap_stage2_native_with_optional_threads(
                    *forward_model,
                    *backward_model,
                    forward_syndromes,
                    backward_syndromes,
                    K,
                    Delta,
                    score_alpha,
                    metric_mode,
                    int_metric_scale,
                    results
                );
            } catch (...) {
                decode_error = std::current_exception();
            }
            Py_END_ALLOW_THREADS
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(results.size()));
        if (list == nullptr) return nullptr;
        for (std::size_t index = 0; index < results.size(); ++index) {
            PyObject* item = stage1_nocap_stage2_payload_to_replay_dict(
                results[index],
                forward_model->n_logical_limbs
            );
            if (item == nullptr) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(index), item);
        }
        return list;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_model_info(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    try {
        NativeModel* model = model_from_capsule(capsule);
        PyObject* dict = PyDict_New();
        if (dict == nullptr) return nullptr;
        if (
            !dict_set_steal(dict, "num_detectors", PyLong_FromLong(model->num_detectors)) ||
            !dict_set_steal(dict, "num_observables", PyLong_FromLong(model->num_observables)) ||
            !dict_set_steal(dict, "n_limbs", PyLong_FromLong(model->n_limbs)) ||
            !dict_set_steal(
                dict,
                "compact_key_limbs",
                PyBool_FromLong((!model->force_full_key && model->n_limbs <= kCompactLimbs) ? 1 : 0)
            ) ||
            !dict_set_steal(
                dict,
                "active4_key",
                PyBool_FromLong((!model->force_full_key && model->n_limbs > kCompactLimbs && model->active4_supported) ? 1 : 0)
            ) ||
            !dict_set_steal(dict, "collect_phase_timing", PyBool_FromLong(model->collect_phase_timing ? 1 : 0)) ||
            !dict_set_steal(dict, "force_full_key", PyBool_FromLong(model->force_full_key ? 1 : 0)) ||
            !dict_set_steal(dict, "columns", PyLong_FromSize_t(model->columns.size()))
        ) {
            Py_DECREF(dict);
            return nullptr;
        }
        return dict;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyObject* py_choice_model_info(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    try {
        ChoiceNativeModel* model = choice_model_from_capsule(capsule);
        PyObject* dict = PyDict_New();
        if (dict == nullptr) return nullptr;
        if (
            !dict_set_steal(dict, "num_detectors", PyLong_FromLong(model->num_detectors)) ||
            !dict_set_steal(dict, "num_observables", PyLong_FromLong(model->num_observables)) ||
            !dict_set_steal(dict, "n_limbs", PyLong_FromLong(model->n_limbs)) ||
            !dict_set_steal(dict, "collect_phase_timing", PyBool_FromLong(model->collect_phase_timing ? 1 : 0)) ||
            !dict_set_steal(dict, "columns", PyLong_FromSize_t(model->columns.size()))
        ) {
            Py_DECREF(dict);
            return nullptr;
        }
        return dict;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"make_model", py_make_model, METH_VARARGS, "Create a native binary frontier model capsule."},
    {"make_choice_model", py_make_choice_model, METH_VARARGS, "Create a native multi-choice frontier model capsule."},
    {"decode", py_decode, METH_VARARGS, "Decode a syndrome using a native binary frontier model."},
    {"decode_choice", py_decode_choice, METH_VARARGS, "Decode a syndrome using a native multi-choice frontier model."},
    {"decode_many", py_decode_many, METH_VARARGS, "Decode a batch of syndromes using a native binary frontier model."},
    {"decode_many_choice", py_decode_many_choice, METH_VARARGS, "Decode a batch of syndromes using a native multi-choice frontier model."},
    {"decode_many_select", py_decode_many_select, METH_VARARGS, "Decode forward/backward batches and return selected committee payloads."},
    {"decode_many_select_compact", py_decode_many_select_compact, METH_VARARGS, "Decode forward/backward batches and return compact selected committee payloads."},
    {"decode_many_select_replay", py_decode_many_select_replay, METH_VARARGS, "Decode forward/backward batches and return flat selected replay payloads."},
    {"model_info", py_model_info, METH_VARARGS, "Return native model metadata."},
    {"choice_model_info", py_choice_model_info, METH_VARARGS, "Return native choice model metadata."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_frontier_native",
    "Native binary frontier engine.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__frontier_native(void) {
    return PyModule_Create(&module);
}
