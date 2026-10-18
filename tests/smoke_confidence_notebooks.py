"""Execute both updated audit notebooks in separate real Jupyter kernels.

Run: MPLCONFIGDIR=/tmp/osfbm-mpl .venv/bin/python tests/smoke_confidence_notebooks.py
Artifacts are saved under result_optimal_stopping/ci_smoke/.
"""
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]


def execute(name, replacement, output, forbid_computation=False):
    notebook = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code' and (
                'cfg = Config(' in cell.source or cell.source.startswith('RUN_ID =')):
            cell.source = replacement
            break
    if forbid_computation:
        notebook.cells.insert(0, nbformat.v4.new_code_cell('''
from osfbm import adapter
from osfbm import confidence, experiment

def forbidden(*args, **kwargs):
    raise AssertionError("Analysis must not simulate or fit")
adapter.simulate_pair = forbidden
adapter.simulate_paths = forbidden
adapter.longstaff_schwartz = forbidden
confidence.restore_policy = forbidden
experiment.train_policy = forbidden
'''))
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=180,
                            resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, output)


def main():
    parent = ROOT / 'result_optimal_stopping' / 'ci_smoke'
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='run_', dir=parent))
    source = directory / 'source'
    execute('03-grid.ipynb', f"""
MODE = "fresh"
RUN_DIR = Path({str(source)!r})
cfg = Config(n_fine=4, n_exercise=2, M_train=256, M_test=512, K=1,
             H_grid=(.3, .5), mu_grid=tuple(np.round(np.arange(-2, 2.01, .25), 6)))
BATCH_SIZE = 128
M_VALIDATION = 128
""", directory / '03-executed.ipynb')
    execute('04-grid-analysis.ipynb', f"""
RUN_ID = "smoke"
ANALYSIS_MODE = "fresh"
H = .3
EPSILON = .3
CONFIDENCE = .9
RUN_DIR = Path({str(source)!r})
""", directory / '04-executed.ipynb', forbid_computation=True)
    for name in ('values.csv', 'boundaries.csv', 'values_H_0.3.png', 'boundaries_ci.png'):
        assert (source / 'analysis' / 'epsilon_0.3' / 'confidence_0.9' / name).is_file(), name
    print(f'Separate-kernel notebook smoke passed: {directory}')


if __name__ == '__main__':
    main()
