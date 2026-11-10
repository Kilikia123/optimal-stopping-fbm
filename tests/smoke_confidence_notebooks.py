"""Execute the training notebook and saved-statistics analysis in separate Jupyter kernels.

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
    notebook = (nbformat.read(ROOT / 'notebooks' / name, as_version=4) if name
                else nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(replacement)]))
    for cell in notebook.cells:
        if cell.cell_type == 'code' and (
                'cfg = Config(' in cell.source or cell.source.startswith('RUN_ID =')):
            cell.source = replacement
            break
    if forbid_computation:
        notebook.cells.insert(0, nbformat.v4.new_code_cell('''
import adapter
from experiments import audit, training

def forbidden(*args, **kwargs):
    raise AssertionError("Analysis must not simulate or fit")
adapter.simulate_pair = forbidden
adapter.simulate_paths = forbidden
adapter.longstaff_schwartz = forbidden
audit.restore_policy = forbidden
training.train_policy = forbidden
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
COLLECT_PAIRED = True
RUN_DIR = Path({str(source)!r})
cfg = Config(n_fine=4, n_exercise=2, M_train=256, M_test=512, K=1,
             H_grid=(.3, .5), mu_grid=tuple(np.round(np.arange(-2, 2.01, .25), 6)))
BATCH_SIZE = 128
M_VALIDATION = 128
""", directory / '03-executed.ipynb')
    execute(None, f"""
from pathlib import Path
from experiments.statistics import load_experiment
from boundary_analysis.reporting import analyze_relative_run
source = Path({str(source)!r})
_, nodes, fixed = load_experiment(source, epsilon=.3, confidence=.9)
assert nodes.complete.all()
assert len(fixed) == 4
analyze_relative_run(source, alpha=.2, confidence=.9,
                     sensitivity=(.1, .2, .3), gain_mode="paired")
""", directory / 'analysis-executed.ipynb', forbid_computation=True)
    for name in ('boundaries.csv', 'diagnostics.csv', 'metadata.json', 'boundaries.png', 'epsilon.png'):
        assert (source / 'analysis' / 'relative_paired' / 'alpha_0.2' / 'confidence_0.9' / name).is_file(), name
    print(f'Separate-kernel notebook smoke passed: {directory}')


if __name__ == '__main__':
    main()
