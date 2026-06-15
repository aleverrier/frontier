from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
UPSTREAM_REPO_URL = "https://github.com/gongaa/SlidingWindowDecoder.git"
UPSTREAM_COMMIT = "05d6b1f478f2b044effdc7477278647dfb99db07"
DEFAULT_UPSTREAM_REPO_DIR = REPO_ROOT / "external" / "SlidingWindowDecoder"
DEFAULT_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "mplcache_betterbeam"
LOCAL_PATCH_VERSION = "better_beam_sliding_window_patch_v10"
BUILD_SENTINEL = ".better_beam_build.json"
PATCH_SENTINEL = ".better_beam_patch.json"


if "MPLCONFIGDIR" not in os.environ:
    DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPLCONFIGDIR)


@dataclass(frozen=True)
class SlidingWindowRepoInfo:
    repo_dir: Path
    repo_url: str
    commit: str
    patch_version: str
    built_with_python: str | None


def _run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str | None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        return proc.stdout.strip()
    return None


def _repo_dirty(repo_dir: Path) -> bool:
    out = _run(["git", "-C", str(repo_dir), "status", "--porcelain"], capture=True)
    return bool(out)


def _has_commit(repo_dir: Path, commit: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    return proc.returncode == 0


def ensure_upstream_repo(
    repo_dir: Path = DEFAULT_UPSTREAM_REPO_DIR,
    *,
    repo_url: str = UPSTREAM_REPO_URL,
    expected_commit: str = UPSTREAM_COMMIT,
) -> SlidingWindowRepoInfo:
    repo_dir = repo_dir.resolve()
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        _run(["git", "clone", repo_url, str(repo_dir)])
    if not (repo_dir / ".git").exists():
        raise RuntimeError(f"{repo_dir} exists but is not a git repository")
    remote_url = _run(["git", "-C", str(repo_dir), "remote", "get-url", "origin"], capture=True) or ""
    if remote_url.rstrip("/") != repo_url.rstrip("/"):
        raise RuntimeError(f"{repo_dir} points at {remote_url}, expected {repo_url}")
    head_before = _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or ""
    if head_before != expected_commit and _repo_dirty(repo_dir):
        raise RuntimeError(f"refusing to change {repo_dir}: repository has local modifications")
    if head_before != expected_commit and not _has_commit(repo_dir, expected_commit):
        _run(["git", "-C", str(repo_dir), "fetch", "origin"])
    if head_before != expected_commit:
        _run(["git", "-C", str(repo_dir), "checkout", "--detach", expected_commit])
    head = _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or ""
    if head != expected_commit:
        raise RuntimeError(f"expected {expected_commit} after checkout, got {head}")
    patch_info = _read_json_if_exists(repo_dir / PATCH_SENTINEL)
    build_info = _read_json_if_exists(repo_dir / BUILD_SENTINEL)
    return SlidingWindowRepoInfo(
        repo_dir=repo_dir,
        repo_url=repo_url,
        commit=head,
        patch_version=str(patch_info.get("patch_version", "")) if isinstance(patch_info, dict) else "",
        built_with_python=str(build_info.get("python", "")) if isinstance(build_info, dict) else None,
    )


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_upstream_text(repo_dir: Path, rel_path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{UPSTREAM_COMMIT}:{rel_path}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def _replace_all(text: str, replacements: list[tuple[str, str]], *, path: Path) -> str:
    out = text
    for old, new in replacements:
        if old in out:
            out = out.replace(old, new)
            continue
        if new in out:
            continue
        raise RuntimeError(f"expected snippet not found in {path}")
    return out


def apply_local_compatibility_patch(repo_dir: Path = DEFAULT_UPSTREAM_REPO_DIR) -> SlidingWindowRepoInfo:
    repo_dir = repo_dir.resolve()
    sentinel_payload = _read_json_if_exists(repo_dir / PATCH_SENTINEL)
    if isinstance(sentinel_payload, dict) and str(sentinel_payload.get("patch_version")) == LOCAL_PATCH_VERSION:
        build_info = _read_json_if_exists(repo_dir / BUILD_SENTINEL)
        return SlidingWindowRepoInfo(
            repo_dir=repo_dir,
            repo_url=UPSTREAM_REPO_URL,
            commit=_run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or "",
            patch_version=LOCAL_PATCH_VERSION,
            built_with_python=str(build_info.get("python", "")) if isinstance(build_info, dict) else None,
        )

    setup_path = repo_dir / "setup.py"
    mod2sparse_cpp = _read_upstream_text(repo_dir, "src/include/mod2sparse.c")
    mod2sparse_cpp = _replace_all(
        mod2sparse_cpp,
        [
            ("    b = chk_alloc(1, sizeof *b);\n", "    b = (mod2block *) chk_alloc(1, sizeof *b);\n"),
            ("  m = chk_alloc (1, sizeof *m);\n", "  m = (mod2sparse *) chk_alloc (1, sizeof *m);\n"),
            ("  m->rows = chk_alloc (n_rows, sizeof *m->rows);\n", "  m->rows = (mod2entry *) chk_alloc (n_rows, sizeof *m->rows);\n"),
            ("  m->cols = chk_alloc (n_cols, sizeof *m->cols);\n", "  m->cols = (mod2entry *) chk_alloc (n_cols, sizeof *m->cols);\n"),
            ("  rinv = chk_alloc(M, sizeof *rinv);\n", "  rinv = (int *) chk_alloc(M, sizeof *rinv);\n"),
            ("  cinv = chk_alloc(N, sizeof *cinv);\n", "  cinv = (int *) chk_alloc(N, sizeof *cinv);\n"),
            ("    acnt = chk_alloc(M+1, sizeof *acnt);\n", "    acnt = (int *) chk_alloc(M+1, sizeof *acnt);\n"),
            ("    rcnt = chk_alloc(M, sizeof *rcnt);\n", "    rcnt = (int *) chk_alloc(M, sizeof *rcnt);\n"),
        ],
        path=repo_dir / "src" / "include" / "mod2sparse_cpp.cpp",
    )
    (repo_dir / "src" / "include" / "mod2sparse_cpp.cpp").write_text(mod2sparse_cpp, encoding="utf-8")

    setup_path.write_text(
        """from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

VERSION = "0.0.0"
(open("src/VERSION", "w+", encoding="utf-8")).write(VERSION)

extension1 = Extension(
    name="src.mod2sparse",
    sources=["src/include/mod2sparse.c", "src/mod2sparse.pyx"],
    include_dirs=[numpy.get_include(), "src/include"],
    extra_compile_args=["-std=c11"],
)

extension2 = Extension(
    name="src.c_util",
    sources=["src/c_util.pyx", "src/include/mod2sparse.c"],
    include_dirs=[numpy.get_include(), "src/include"],
    extra_compile_args=["-std=c11"],
)

extension3 = Extension(
    name="src.bp_guessing_decoder",
    sources=["src/bp_guessing_decoder.pyx", "src/include/mod2sparse_cpp.cpp", "src/include/bpgd.cpp"],
    language="c++",
    include_dirs=[numpy.get_include(), "src/include"],
    extra_compile_args=["-std=c++14"],
)

extension4 = Extension(
    name="src.osd_window",
    sources=["src/osd_window.pyx", "src/include/mod2sparse_cpp.cpp", "src/include/mod2sparse_extra.cpp", "src/include/bpgd.cpp"],
    language="c++",
    include_dirs=[numpy.get_include(), "src/include"],
    extra_compile_args=["-std=c++14"],
)

setup(
    version=VERSION,
    ext_modules=cythonize(
        [extension1, extension2, extension3, extension4],
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "embedsignature": True,
            "language_level": 3,
        },
    ),
)
""",
        encoding="utf-8",
    )

    c_util_pxd = repo_dir / "src" / "c_util.pxd"
    c_util_pxd.write_text(
        """#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libc.stdlib cimport malloc, calloc,free
import numpy as np
cimport numpy as np
cimport cython

cdef char* numpy2char(np_array, char* char_array)
cdef char* spmatrix2char(matrix, char* char_array)
cdef int* numpy2int(np_array, int* int_array)
cdef double* numpy2double(np.ndarray[np.float64_t, ndim=1] np_array, double* double_array)
cdef np.ndarray[np.int64_t, ndim=1] char2numpy(char* char_array, int n)
cdef np.ndarray[np.int64_t, ndim=2] stackchar2numpy(char* arr1, char* arr2, int n)
cdef np.ndarray[np.float64_t, ndim=1] double2numpy(double* char_array, int n)
""",
        encoding="utf-8",
    )

    c_util_pyx = repo_dir / "src" / "c_util.pyx"
    c_util_pyx.write_text(
        """#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
from scipy.sparse import spmatrix

cdef char* numpy2char(np_array, char* char_array):
    cdef int n = np_array.shape[0]
    for i in range(n):
        char_array[i] = np_array[i]
    return char_array

cdef char* spmatrix2char(matrix, char* char_array):
    cdef int n = matrix.shape[1]
    for i in range(n):
        char_array[i] = 0
    for i, j in zip(*matrix.nonzero()):
        char_array[j] = 1
    return char_array

cdef int* numpy2int(np_array, int* int_array):
    cdef int n = np_array.shape[0]
    for i in range(n):
        int_array[i] = np_array[i]
    return int_array

cdef double* numpy2double(np.ndarray[np.float64_t, ndim=1] np_array, double* double_array):
    cdef int n = np_array.shape[0]
    for i in range(n):
        double_array[i] = np_array[i]
    return double_array

cdef np.ndarray[np.int64_t, ndim=1] char2numpy(char* char_array, int n):
    cdef np.ndarray[np.int64_t, ndim=1] np_array = np.zeros(n, dtype=np.int64)
    for i in range(n):
        np_array[i] = char_array[i]
    return np_array

cdef np.ndarray[np.float64_t, ndim=1] double2numpy(double* char_array, int n):
    cdef np.ndarray[np.float64_t, ndim=1] np_array = np.zeros(n, dtype=np.float64)
    for i in range(n):
        np_array[i] = char_array[i]
    return np_array

cdef np.ndarray[np.int64_t, ndim=2] stackchar2numpy(char* arr1, char* arr2, int n):
    cdef np.ndarray[np.int64_t, ndim=2] np_array = np.zeros((2, n), dtype=np.int64)
    for i in range(n):
        np_array[0, i] = arr1[i]
        np_array[1, i] = arr2[i]
    return np_array
""",
        encoding="utf-8",
    )

    init_path = repo_dir / "src" / "__init__.py"
    init_path.write_text(
        """import os
from .bp_guessing_decoder import bpgdg_decoder, bpgd_decoder, bp_history_decoder
from .osd_window import osd_window

try:
    from .bp4_osd import bp4_osd
except Exception:  # optional in better-beam local build
    bp4_osd = None

from . import __file__


def get_include():
    path = os.path.dirname(__file__)
    return path


f = open(get_include() + "/VERSION")
__version__ = f.read()
f.close()
""",
        encoding="utf-8",
    )

    for rel_path in (
        "src/bp_guessing_decoder.pxd",
        "src/bp_guessing_decoder.pyx",
        "src/osd_window.pxd",
        "src/osd_window.pyx",
    ):
        path = repo_dir / rel_path
        text = _read_upstream_text(repo_dir, rel_path)
        text = text.replace("np.int_t", "np.int64_t")
        text = text.replace("np.float_t", "np.float64_t")
        text = text.replace("long(2 ** self.osd_order)", "int(2 ** self.osd_order)")
        text = text.replace("np.zeros((self.n, self.history_length))", "np.zeros((self.n, self.history_length), dtype=np.float64)")
        if rel_path == "src/bp_guessing_decoder.pyx":
            text = _replace_all(
                text,
                [
                    (
                        """    @property\n    def converge(self):\n        return self.converge\n\ncdef class bpgdg_decoder(bp_history_decoder):\n""",
                        """    @property\n    def bp_iteration(self):\n        return self.bp_iteration\n\n    @property\n    def converge(self):\n        return self.converge\n\ncdef class bpgdg_decoder(bp_history_decoder):\n""",
                    ),
                    (
                        """    cpdef np.ndarray[np.int64_t, ndim=1] decode(self, input_vector):\n        cdef int input_length = input_vector.shape[0]\n\n        if input_length == self.m:\n""",
                        """    cpdef np.ndarray[np.int64_t, ndim=1] decode(self, input_vector):\n        cdef int input_length = input_vector.shape[0]\n\n        self.bp_iteration = 0\n        self.converge = False\n        if input_length == self.m:\n""",
                    ),
                    (
                        """        self.bpgd_main_thread.do_work(self.H, self.cols, self.channel_llr, self.synd)\n""",
                        """        self.bp_iteration += self.max_iter_per_step * self.max_step\n        self.bpgd_main_thread.do_work(self.H, self.cols, self.channel_llr, self.synd)\n""",
                    ),
                    (
                        """        self.bp_iteration = 0 \n        self.min_converge_depth = self.max_step\n""",
                        """        self.min_converge_depth = self.max_step\n""",
                    ),
                    (
                        """        for current_depth in range(self.max_step):\n            temp_converge = self.bpgd.min_sum_log()\n""",
                        """        for current_depth in range(self.max_step):\n            self.bp_iteration += self.max_iter_per_step\n            temp_converge = self.bpgd.min_sum_log()\n""",
                    ),
                    (
                        """            for j in range(self.max_side_branch_step):\n                current_depth = self.alt_depth_stack[i] + j\n                temp_converge = self.bpgd.min_sum_log()\n""",
                        """            for j in range(self.max_side_branch_step):\n                current_depth = self.alt_depth_stack[i] + j\n                self.bp_iteration += self.max_iter_per_step\n                temp_converge = self.bpgd.min_sum_log()\n""",
                    ),
                ],
                path=path,
            )
        path.write_text(text, encoding="utf-8")

    mod2sparse_h = repo_dir / "src" / "include" / "mod2sparse.h"
    mod2sparse_h_text = _read_upstream_text(repo_dir, "src/include/mod2sparse.h")
    header_start = "#define MOD2SPARSE_H\n"
    header_end = "#endif /* MOD2SPARSE_H */"
    if header_start not in mod2sparse_h_text or header_end not in mod2sparse_h_text:
        raise RuntimeError(f"unexpected header layout in {mod2sparse_h}")
    mod2sparse_h_text = mod2sparse_h_text.replace(
        header_start,
        '#define MOD2SPARSE_H\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n',
        1,
    )
    mod2sparse_h_text = mod2sparse_h_text.replace(
        header_end,
        '#ifdef __cplusplus\n}\n#endif\n\n#endif /* MOD2SPARSE_H */',
        1,
    )
    mod2sparse_h.write_text(mod2sparse_h_text, encoding="utf-8")

    bpgd_cpp = repo_dir / "src" / "include" / "bpgd.cpp"
    bpgd_text = _read_upstream_text(repo_dir, "src/include/bpgd.cpp")
    bpgd_text = _replace_all(
        bpgd_text,
        [
            (
                "#include <sched.h>\n#include <pthread.h>\n",
                "#include <pthread.h>\n#ifdef __linux__\n#include <sched.h>\n#endif\n",
            ),
            (
                """void set_affinity(std::thread& th, int core_id) {\n    cpu_set_t cpuset;\n    CPU_ZERO(&cpuset);\n    CPU_SET(core_id, &cpuset);\n    int rc = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);\n    if (rc != 0) {\n        std::cerr << \"Error setting thread affinity: \" << rc << std::endl;\n    }\n}\n""",
                """void set_affinity(std::thread& th, int core_id) {\n#ifdef __linux__\n    cpu_set_t cpuset;\n    CPU_ZERO(&cpuset);\n    CPU_SET(core_id, &cpuset);\n    int rc = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);\n    if (rc != 0) {\n        std::cerr << \"Error setting thread affinity: \" << rc << std::endl;\n    }\n#else\n    (void)th;\n    (void)core_id;\n#endif\n}\n""",
            ),
            (
                """    cpu_set_t cpuset;\n    CPU_ZERO(&cpuset);\n    CPU_SET(num_tree_threads, &cpuset); // main thread is assigned to core 7\n    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);\n    if (rc != 0) cerr << \"Error setting thread affinity: \" << rc << endl;\n""",
                """#ifdef __linux__\n    cpu_set_t cpuset;\n    CPU_ZERO(&cpuset);\n    CPU_SET(num_tree_threads, &cpuset);\n    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);\n    if (rc != 0) cerr << \"Error setting thread affinity: \" << rc << endl;\n#endif\n""",
            ),
        ],
        path=bpgd_cpp,
    )
    bpgd_cpp.write_text(bpgd_text, encoding="utf-8")

    _write_json(
        repo_dir / PATCH_SENTINEL,
        {
            "patch_version": LOCAL_PATCH_VERSION,
            "upstream_commit": _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or "",
        },
    )
    build_info = _read_json_if_exists(repo_dir / BUILD_SENTINEL)
    return SlidingWindowRepoInfo(
        repo_dir=repo_dir,
        repo_url=UPSTREAM_REPO_URL,
        commit=_run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or "",
        patch_version=LOCAL_PATCH_VERSION,
        built_with_python=str(build_info.get("python", "")) if isinstance(build_info, dict) else None,
    )


def build_upstream_extensions(
    repo_dir: Path = DEFAULT_UPSTREAM_REPO_DIR,
    *,
    python_bin: str | None = None,
    force: bool = False,
) -> SlidingWindowRepoInfo:
    repo_dir = repo_dir.resolve()
    python_bin = str(python_bin or sys.executable)
    sentinel_payload = _read_json_if_exists(repo_dir / BUILD_SENTINEL)
    patch_payload = _read_json_if_exists(repo_dir / PATCH_SENTINEL)
    if (
        not force
        and isinstance(sentinel_payload, dict)
        and str(sentinel_payload.get("python")) == python_bin
        and str(sentinel_payload.get("patch_version")) == LOCAL_PATCH_VERSION
    ):
        return SlidingWindowRepoInfo(
            repo_dir=repo_dir,
            repo_url=UPSTREAM_REPO_URL,
            commit=_run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or "",
            patch_version=str(patch_payload.get("patch_version", "")) if isinstance(patch_payload, dict) else "",
            built_with_python=python_bin,
        )
    _run([python_bin, "setup.py", "build_ext", "--inplace", "--force"], cwd=repo_dir)
    _write_json(
        repo_dir / BUILD_SENTINEL,
        {
            "python": python_bin,
            "patch_version": LOCAL_PATCH_VERSION,
            "upstream_commit": _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or "",
        },
    )
    return SlidingWindowRepoInfo(
        repo_dir=repo_dir,
        repo_url=UPSTREAM_REPO_URL,
        commit=_run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture=True) or "",
        patch_version=LOCAL_PATCH_VERSION,
        built_with_python=python_bin,
    )


def ensure_upstream_ready(
    repo_dir: Path = DEFAULT_UPSTREAM_REPO_DIR,
    *,
    python_bin: str | None = None,
) -> SlidingWindowRepoInfo:
    info = ensure_upstream_repo(repo_dir)
    info = apply_local_compatibility_patch(info.repo_dir)
    return build_upstream_extensions(info.repo_dir, python_bin=python_bin)


def prepend_repo_to_syspath(repo_dir: Path = DEFAULT_UPSTREAM_REPO_DIR) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
