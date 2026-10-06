import pytest

from nanochat.dataset import list_parquet_files, parquets_iter_batched, resolve_parquet_data_dir
from nanochat.dataloader import _document_batches


def test_parquet_directory_override(monkeypatch, tmp_path):
    monkeypatch.setenv('LOCAL_PARQUET_DIR', str(tmp_path / 'env'))
    assert resolve_parquet_data_dir() == str(tmp_path / 'env')
    assert resolve_parquet_data_dir(str(tmp_path / 'explicit')) == str(tmp_path / 'explicit')


def test_empty_directory_fails_without_hanging(tmp_path):
    with pytest.raises(FileNotFoundError, match='No parquet shards'):
        list_parquet_files(str(tmp_path))


def test_one_shard_cannot_be_both_train_and_validation(tmp_path):
    (tmp_path / 'shard_00000.parquet').touch()
    with pytest.raises(ValueError, match='separate validation'):
        next(parquets_iter_batched('train', data_dir=str(tmp_path)))


def test_training_dataloader_rejects_missing_training_shard(tmp_path):
    (tmp_path / 'shard_00000.parquet').touch()
    with pytest.raises(ValueError, match='separate validation'):
        next(_document_batches('train', None, 1, data_dir=str(tmp_path)))
