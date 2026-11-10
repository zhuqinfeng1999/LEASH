"""Production factories used by both compiled runner and CPU fixtures."""
from . import bindings
from .hooks import toolkit_module

_compiled = toolkit_module()
E1MotionReviewer = _compiled.E1MotionReviewer
PublicWorkspaceFactory = bindings.PublicWorkspaceFactory


class E1GatedToolkit(_compiled.E1GatedToolkit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._public_proposal = bindings.proposal

    def operation_scope(self, *args, **kwargs):
        raise ValueError('No pick macro is registered in this common protocol')


class PlainRPentToolkit:
    """No added pick skill, no alternate action path, same FaultStop base."""
    def __init__(self, base, *, public_packet):
        self.base = base
        self.events = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        specs = self.base.get_tools_spec()
        if any(spec['name'] == 'pick' for spec in specs):
            raise RuntimeError('unexpected pick macro in ordinary RPent protocol')
        return specs

    def execute_tool(self, name, arguments):
        if name == 'pick':
            raise ValueError('pick macro is not part of this explicit protocol')
        return self.base.execute_tool(name, arguments)
