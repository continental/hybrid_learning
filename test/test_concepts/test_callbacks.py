"""Tests for callbacks."""

#  Copyright (c) 2022 Continental Automotive GmbH
import logging
import os
from typing import Dict, Any
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hybrid_learning.concepts.train_eval import callbacks


class TestCsvLoggingCallback:
    """Test the figure logging."""

    @pytest.fixture
    def fig_data(self) -> pd.DataFrame:
        """A dataframe as it should look like for a figure with 1 axis and 2 lines."""
        return pd.DataFrame({'xcol': np.arange(0, 20),
                             'ycol': np.arange(100, 120),
                             'line': ['_line0'] * 10 + ['_line1'] * 10})

    @pytest.fixture
    def fig(self, fig_data: pd.DataFrame) -> plt.Figure:
        """Figure with 1 axis but 2 lines."""
        _, ax = plt.subplots()
        line0_data = fig_data[fig_data.line.str.contains('0')]
        line1_data = fig_data[fig_data.line.str.contains('1')]
        ax.plot(line0_data.iloc[:, 0], line0_data.iloc[:, 1])
        ax.plot(line1_data.iloc[:, 0], line1_data.iloc[:, 1])
        ax: plt.Axes = plt.gca()
        ax.set_xlabel(fig_data.columns[0])
        ax.set_ylabel(fig_data.columns[1])
        yield plt.gcf()
        plt.close()

    @pytest.fixture
    def fig_cb(self, tmp_path):
        """Simple figure logging callback."""
        return callbacks.CsvLoggingCallback(str(tmp_path))

    @pytest.mark.parametrize('kwargs,out', [
        ({}, ''),
        (dict(log_prefix='log-prefix'), 'log-prefix'),
        (dict(run=0), 'run0'),
        (dict(epoch=1), 'epoch1'),
        (dict(batch=2), 'batch2'),
        (dict(log_prefix='D', run='R'), 'D_runR'),
        (dict(log_prefix='D', run='R', epoch='E'), 'D_runR/epochE'),
        (dict(log_prefix='D', run='R', epoch='E', batch='B'), 'D_runR/epochE/batchB'),
        (dict(run='R', epoch='E', batch='B'), 'runR/epochE/batchB'),
        (dict(run='R', epoch='E'), 'runR/epochE'),
    ])
    def test_to_descriptor(self, fig_cb, kwargs: Dict[str, Any], out: str):
        out = os.path.normpath(out) if out else out
        assert fig_cb.to_descriptor(**kwargs) == out
        assert fig_cb.from_descriptor(fig_cb.to_descriptor(**kwargs)) == \
               {k: str(v) for k, v in kwargs.items()}
        if len(kwargs) > 0:
            assert fig_cb.file_path_for(kpi_name='some_kpi', **kwargs) == \
                os.path.join(fig_cb.log_dir, out, 'some_kpi.csv')
        else:
            assert fig_cb.file_path_for(kpi_name='some_kpi', **kwargs, default_desc='blub') == \
                   os.path.join(fig_cb.log_dir, 'blub', 'some_kpi.csv')
            with pytest.raises(ValueError):
                fig_cb.file_path_for(kpi_name='some_kpi', **kwargs)

    def test_after_epoch_eval(self, fig, fig_data, fig_cb):
        metric_name = f'some_metric_after_epoch_eval'
        other_metric_name = 'some_other_after_epoch_eval'
        desc = 'some_desc_after_epoch_eval'
        run_type = 'test'
        assert isinstance(fig, plt.Figure)
        self.assert_dataframes_equal(fig_cb._fig_to_pd(fig), fig_data)
        fig_cb.after_epoch_eval(kpi_val=pd.Series({metric_name: fig, other_metric_name: 1}),
                                log_prefix=desc, run_type=run_type)

        # File created
        for kpi_name, val in ((metric_name, fig_data),
                              (f'{run_type}_{callbacks.CsvLoggingCallback.OTHER_KPI_NAME}',
                               pd.DataFrame({other_metric_name: [1]}))):
            # File exists
            file_path: str = fig_cb.file_path_for(log_prefix=desc, kpi_name=kpi_name)
            assert os.path.exists(file_path)
            assert os.path.isfile(file_path)
            # Content correct
            saved = pd.read_csv(file_path).infer_objects()
            if 'index' in saved.columns:
                saved.set_index('index', inplace=True, drop=True)
            print("DEBUG", file_path)
            self.assert_dataframes_equal(saved, val)
            # Cleanup
            os.path.isfile(file_path)

    @staticmethod
    def assert_dataframes_equal(got, expected):
        """Check equality of two pandas DataFrames. Raise assertion error if unequal."""
        assert sorted(got.index) == sorted(expected.index), \
            "Got:\n{}\nExpected:\n{}".format(got, expected)
        assert sorted(got.columns) == sorted(expected.columns), \
            "Got:\n{}\nExpected:\n{}".format(got, expected)
        assert (got == expected).to_numpy().all(), \
            "Got:\n{}\nExpected:\n{}".format(got, expected)

    def test_after_epoch_train(self, fig, fig_data, fig_cb):
        metric_name = 'some_metric_after_epoch_train'
        other_metric_name = 'some_other_after_epoch_train'
        desc = 'some_desc_after_epoch_train'
        run_type = 'training'
        assert isinstance(fig, plt.Figure)
        self.assert_dataframes_equal(fig_cb._fig_to_pd(fig), fig_data)
        fig_cb.after_epoch_train(kpi_train=pd.DataFrame({metric_name: [None, None, fig],
                                                         other_metric_name: [1, 2, 3]}),
                                 log_prefix=desc, run_type=run_type)
        # File created
        for kpi_name, val in ((metric_name, fig_data),
                              (f'{run_type}_{callbacks.CsvLoggingCallback.OTHER_KPI_NAME}',
                               pd.DataFrame({other_metric_name: [1, 2, 3]}))):
            # File exists
            file_path: str = fig_cb.file_path_for(log_prefix=desc, kpi_name=kpi_name)
            assert os.path.exists(file_path)
            assert os.path.isfile(file_path)
            # Content correct
            saved = pd.read_csv(file_path).infer_objects()
            if 'index' in saved.columns:
                saved.set_index('index', inplace=True, drop=True)
            self.assert_dataframes_equal(saved, val)

    @pytest.mark.parametrize('use_abs_paths', (True, False))
    def test_metrics_data(self, fig, fig_data, fig_cb, use_abs_paths: bool):
        other_metric_name = 'some_other'
        desc = 'some_desc'
        run_type = 'training'
        epoch, run = 42, 3
        assert isinstance(fig, plt.Figure)
        fig_cb.after_epoch_train(kpi_train=pd.DataFrame({'some_metric0': [None, None, fig],
                                                         'some_metric1': [None, None, fig],
                                                         other_metric_name: [1, 2, 3]}),
                                 log_prefix=desc, epoch=epoch, run=run, run_type=run_type)
        metrics_info = fig_cb.file_paths_in(fig_cb.log_dir, use_abs_paths=use_abs_paths)
        assert isinstance(metrics_info, list)
        assert len(metrics_info) == 3
        for info in metrics_info:
            assert 'log_prefix' in info, info
            assert info['log_prefix'] == desc, info
            assert 'epoch' in info, info
            assert info['epoch'] == str(epoch), info
            assert 'run' in info, info
            assert info['run'] == str(run), info
            assert 'file_path' in info
            assert os.path.exists(info['file_path'] if use_abs_paths else
                                  os.path.join(fig_cb.log_dir, info['file_path']))
            assert 'kpi_name' in info

        # File created
        metrics_info_map = {info['kpi_name']: info for info in metrics_info}
        if not use_abs_paths:
            for info in metrics_info_map.values():
                info['file_path'] = os.path.join(fig_cb.log_dir, info['file_path'])
        assert sorted(metrics_info_map.keys()) == \
               sorted(['some_metric0', 'some_metric1',
                       f'{run_type}_{callbacks.CsvLoggingCallback.OTHER_KPI_NAME}'])
        for m_name, val in (('some_metric0', fig_data),
                            ('some_metric1', fig_data),
                            (f'{run_type}_{callbacks.CsvLoggingCallback.OTHER_KPI_NAME}',
                             pd.DataFrame({other_metric_name: [1, 2, 3]}))):
            saved = fig_cb.from_csv(**metrics_info_map[m_name])
            assert (saved == val).to_numpy().all(), \
                "Saved:\n{}\nExpected:\n{}".format(saved, val)


class TestLoggingCallback:
    @pytest.mark.parametrize("batches,log_per_batch,call_count",
                             [(7, 1, 7), (7, True, 7),
                              (7, 0, 0), (7, False, 0), (7, -1, 0),
                              (7, 3, 3),
                              (6, 3, 2),
                              (6, 2, 3), ])
    def test_log_per_batch(self, batches: int, log_per_batch: int, call_count: int):
        """Check that the call count is correct according to log_per_batch setting."""
        logger = logging.getLogger()
        kpi_train: pd.DataFrame = pd.DataFrame({"set_iou": list(range(batches))})
        with mock.patch.object(logger, 'log', return_value=None) as log_method:
            log_cb: callbacks.LoggingCallback = callbacks.LoggingCallback(logger=logger, log_per_batch=log_per_batch)
            for i in range(batches):
                log_cb.after_batch_train(kpi_train=kpi_train, batch=i, batches=batches)
        assert log_method.call_count == call_count
