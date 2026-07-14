from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_uses_large_44k_v2_by_default():
    source = (ROOT / 'demo.py').read_text()

    assert "default='large_44k_v2'" in source


def test_gradio_delegates_model_validation_to_download_helper():
    source = (ROOT / 'gradio_demo.py').read_text()

    assert 'if not model.model_path.exists()' not in source
    assert '    model.download_if_needed()' in source
