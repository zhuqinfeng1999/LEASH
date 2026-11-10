"""Opt-in offline diagnostic classification; no shared source/default mutation."""
PAD_NAMES=frozenset({'gripper0_right_finger1_pad_collision','gripper0_right_finger2_pad_collision'})
MATERIAL='foil_sandwich_1_foil_skin_collision'
def classify(row):
 names=tuple(row['geoms'])
 if len(names)!=2:raise ValueError('Exact contact pair required')
 dist=float(row['dist_m'])
 if not __import__('math').isfinite(dist):raise ValueError('Nonfinite contact')
 softpad=(names[0]in PAD_NAMES and names[1]==MATERIAL)or(names[1]in PAD_NAMES and names[0]==MATERIAL)
 return {'actual_pair':list(names),'dist_m':dist,'ordinary_pad_material_contact':softpad,'non_grasp_hard_stop':bool(dist<-.001 and not softpad)}
def bind_actual_compiled_pads(geometry):
 """Require exact two known actual boxes/soft parameters before exception use."""
 for name in PAD_NAMES:
  if name not in geometry:raise ValueError('Required actual compiled pad missing')
  row=geometry[name]
  if row['geom_type']!=6 or row['geom_size']!=[.008,.004,.008] or row['geom_solref']!=[.01,.5] or row['geom_contype']!=1 or row['geom_conaffinity']!=1:
   raise ValueError('Compiled pad identity/soft-contact parameters changed')
 return classify
