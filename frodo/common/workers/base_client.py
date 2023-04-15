import abc
from frodo.common.workers import worker


class BaseClient(worker):
    def __init__(self, ID, state,
                 config, model, strategy, mode, address) -> None:
        super(worker, self).__init__(ID, state,
                                     config, model, strategy, mode, address)
        self.msg_handlers = dict()
        self.msh_handlers_str = dict()

    def register_handlers(self, msg_type, callback_func, send_msg=[None]):
        self.msg_handlers[msg_type] = callback_func
        self.msg_handlers_str[msg_type] = (callback_func.__name__, send_msg)

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in, [
            'assign_client_id', 'ask_for_join_in_info', 'address', 'model_para'
        ])
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in,
                               ['address', 'model_para'])
        self.register_handlers('model_para', self.callback_funcs_model_para,
                               ['model_para', 'evaluate', 'finish'])
        self.register_handlers('metrics', self.callback_funcs_for_metrics,
                               ['converged'])

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_model_para(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_join_in(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_metrics(self, message):
        raise NotImplementedError
