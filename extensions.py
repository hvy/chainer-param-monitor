from chainer import reporter
from chainer.training import extension
import statistics


def prefix_statistic_keys(key_prefix, stats):
    for key in list(stats.keys()):
        stats['{}/{}'.format(key_prefix, key)] = stats.pop(key)


class LinkMonitor(extension.Extension):

    default_name = 'link_monitor'
    invoke_before_training = True
    priority = extension.PRIORITY_WRITER

    def __init__(self, links, sparsity=True, sparsity_include_bias=True,
                 monitor_targets=(('W', 'data'),
                                  ('W', 'grad'),
                                  ('b', 'data'),
                                  ('b', 'grad')),
                 prefix=None):

        if not isinstance(links, (tuple, list)):
            links = links,

        self._links = links
        self._sparsity = sparsity
        self._sparsity_include_bias=sparsity_include_bias
        self._monitor_targets = monitor_targets
        self._prefix = prefix
        self._reporter = reporter.Reporter()

    def __call__(self, trainer):
        with self._reporter:
            summary = reporter.DictSummary()
            observation = {}

            with reporter.report_scope(observation):
                for link in self._links:
                    for targets in self._monitor_targets:
                        summary.add(self.get_statistics(link, *targets))
                    if self._sparsity:
                        summary.add(self.get_sparsity(link))

            summary = summary.compute_mean()

        reporter.report(summary)

    def get_statistics(self, link, param_name, attr_name):

        stats = statistics.get_statistics(link, param_name, attr_name)

        if self._prefix is not None:
            prefix_statistic_keys(self._prefix, stats)

        return stats

    def get_sparsity(self, link):

        stats = statistics.get_sparsity(link, self._sparsity_include_bias)

        if self._prefix is not None:
            prefix_statistic_keys(self._prefix, stats)

        return stats
