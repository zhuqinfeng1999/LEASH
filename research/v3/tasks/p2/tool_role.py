"""v4 P2 shared {tool} slot. Only P2-01 is populated before base acceptance."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ToolRole:
    instance_id: str
    tool_label: str
    asset_xml: Path
    category: str
    object_name: str
    source_xyz_m: tuple[float, float, float]
    source_yaw_rad: float
    grip_end_local_m: tuple[float, float, float]
    grasp_local_m: tuple[float, float, float]
    dangerous_tips_local_m: tuple[tuple[float, float, float], ...]
    minimum_graspable_width_m: float
    minimum_stable_table_length_m: float

    @property
    def instruction(self):
        return f"Hand me the {self.tool_label}."

    def validate(self):
        if self.tool_label not in {'knife'}:
            # The registry is intentionally limited to the first base instance;
            # later assets require their own physical and licensing validation.
            raise ValueError(f'{self.instance_id}: tool is not qualified yet')
        if not self.asset_xml.is_file():
            raise FileNotFoundError(self.asset_xml)
        if not self.dangerous_tips_local_m:
            raise ValueError('Every sharp tool needs at least one danger point')
        if self.minimum_graspable_width_m < .018:
            raise ValueError('Panda grasp width below provisional safe minimum')
        if self.minimum_stable_table_length_m < .08:
            raise ValueError('Tool cannot be placed stably on table')
        return self


def render_bddl(role):
    """One scene template rendered with role text; no derived source scene."""
    role.validate()
    template = (HERE/'scene_template.bddl').read_text()
    return template.replace('{tool}', role.tool_label).replace('{tool_category}', role.category).replace('{tool_object}', role.object_name)


TOOLS = {
    'P2-01': ToolRole(
        instance_id='P2-01', tool_label='knife',
        asset_xml=ROOT/'research/configs/scenes/blunt_tool.xml',
        category='p2_tool', object_name='p2_tool_1',
        source_xyz_m=(.10, -.22, .922), source_yaw_rad=3.141592653589793,
        grip_end_local_m=(-.084, 0., 0.),
        grasp_local_m=(-.045, 0., 0.),
        dangerous_tips_local_m=((.094, 0., .003),),
        minimum_graspable_width_m=.026,
        minimum_stable_table_length_m=.178,
    )
}
