# Copyright 2025 Sony AI

class ValidationMonitor(object):
    def __init__(self, writer):
        self._writer = writer

    def add(self, i, val_results):
        all_test_metric = val_results[0]
        other_scores = all_test_metric[5]
        val_loss = val_results[1]

        self._writer.add_scalar('Metrics/1_ER-LD', all_test_metric[0], i)
        self._writer.add_scalar('Metrics/2_F-LD', all_test_metric[1], i)
        self._writer.add_scalar('Metrics/3_LE-CD', all_test_metric[2], i)
        self._writer.add_scalar('Metrics/4_LR-CD', all_test_metric[3], i)
        self._writer.add_scalar('Metrics/0_SELD-error', all_test_metric[4], i)

        self._writer.add_scalar('MetricsDetail/1_ER-Del-LD', other_scores[0], i)
        self._writer.add_scalar('MetricsDetail/1_ER-Ins-LD', other_scores[1], i)
        self._writer.add_scalar('MetricsDetail/1_ER-Sub-LD', other_scores[2], i)
        self._writer.add_scalar('MetricsDetail/2_Precision-LD', other_scores[3], i)
        self._writer.add_scalar('MetricsDetail/2_Recall-LD', other_scores[4], i)
        self._writer.add_scalar('MetricsDetail/4_F-CD', other_scores[5], i)
        self._writer.add_scalar('MetricsDetail/4_Precision-CD', other_scores[6], i)

        self._writer.add_scalar('Loss/val', val_loss, i)
