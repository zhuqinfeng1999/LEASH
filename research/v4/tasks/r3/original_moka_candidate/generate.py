#!/usr/bin/env python3
"""Deterministic, original geometry and textures for an unbranded moka prop.

Unit: metres. Object origin and +Y handle convention match the provisional
LIBERO pot. This script needs only NumPy, Pillow and Matplotlib for previews.
It does not read or transform any third-party model or texture.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(19850417)


class Mesh:
    def __init__(self, name, material, base_color):
        self.name, self.material, self.base_color = name, material, base_color
        self.v, self.uv, self.f = [], [], []

    def vert(self, xyz, uv=(0, 0)):
        self.v.append(tuple(map(float, xyz)))
        self.uv.append(tuple(map(float, uv)))
        return len(self.v)

    def tri(self, a, b, c):
        self.f.append((a, b, c))

    def quad(self, a, b, c, d):
        self.tri(a, b, c)
        self.tri(a, c, d)

    def obj(self):
        path = OUT / f"{self.name}.obj"
        with path.open("w") as out:
            out.write("# Original SafeVLA geometry; units metre; source generate.py\n")
            out.write("mtllib moka_materials.mtl\n")
            out.write(f"o {self.name}\n")
            for v in self.v:
                out.write("v %.8f %.8f %.8f\n" % v)
            for uv in self.uv:
                out.write("vt %.6f %.6f\n" % uv)
            out.write(f"usemtl {self.material}\n")
            for a, b, c in self.f:
                out.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        return path


meshes = []


def lathe(name, material, color, profile, sides=64, facet_strength=0.0, facet_count=8, phase=0.0):
    """Lathe radial profile, optionally blending a true regular polygon surface."""
    m = Mesh(name, material, color)
    for j, (r, z) in enumerate(profile):
        for i in range(sides + 1):
            a = i / sides * 2 * math.pi + phase
            period = 2 * math.pi / facet_count
            delta = (a + period/2) % period - period/2
            polygon = math.cos(math.pi/facet_count) / math.cos(delta)
            rr = r * (1 + facet_strength * (polygon - 1))
            m.vert((rr * math.cos(a), rr * math.sin(a), z),
                   (i / sides, j / (len(profile) - 1)))
    for j in range(len(profile) - 1):
        for i in range(sides):
            a = j * (sides + 1) + i + 1
            b = a + 1
            c = b + sides + 1
            d = a + sides + 1
            m.quad(a, b, c, d)
    meshes.append(m)
    return m


def swept_tube(name, material, color, path, radius_x, radius_y, sides=16):
    """Closed, softly curved oval polymer grip with a Catmull-Rom centerline."""
    m = Mesh(name, material, color)
    control = [np.asarray(p, dtype=float) for p in path]
    smooth=[]
    for j in range(len(control)-1):
        p0=control[max(j-1,0)]; p1=control[j]
        p2=control[j+1]; p3=control[min(j+2,len(control)-1)]
        for k in range(5):
            t=k/5
            smooth.append(.5*((2*p1)+(-p0+p2)*t+
                          (2*p0-5*p1+4*p2-p3)*t*t+
                          (-p0+3*p1-3*p2+p3)*t*t*t))
    smooth.append(control[-1])
    for j, p in enumerate(smooth):
        tangent = smooth[min(j + 1, len(smooth) - 1)] - smooth[max(j - 1, 0)]
        tangent /= np.linalg.norm(tangent)
        reference = (1,0,0) if abs(tangent[0]) < .9 else (0,1,0)
        across = np.cross(tangent, reference)
        across /= np.linalg.norm(across)
        for i in range(sides + 1):
            a = i * 2 * math.pi / sides
            m.vert(p + np.array([1., 0., 0.]) * radius_x * math.cos(a)
                   + across * radius_y * math.sin(a),
                   (i / sides, j / (len(smooth) - 1)))
    for j in range(len(smooth) - 1):
        for i in range(sides):
            a = j * (sides + 1) + i + 1
            m.quad(a, a + 1, a + sides + 2, a + sides + 1)
    meshes.append(m)
    return m


def box(name, material, color, center, half):
    m = Mesh(name, material, color)
    x, y, z = center
    sx, sy, sz = half
    v = [m.vert((x + i * sx, y + j * sy, z + k * sz), ((i + 1) / 2, (j + 1) / 2))
         for i, j, k in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                         (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
    for a,b,c,d in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),
                    (2,3,7,6),(3,0,4,7)]:
        m.quad(v[a],v[b],v[c],v[d])
    meshes.append(m)
    return m


def spout():
    """Shallow, open pouring beak formed by a thin V-channel and rolled lip."""
    m=Mesh('pour_spout','bright_steel',(.88,.88,.84))
    # Each row is y, base height, outer half-width, and wall height. An open
    # five-point U is lofted forward from the upper chamber, not a solid wedge.
    sections=[(-.033,.055,.012,.008),(-.041,.057,.012,.011),
              (-.050,.058,.014,.013),(-.056,.060,.013,.011)]
    for j,(y,base,width,rise) in enumerate(sections):
        for k,frac in enumerate([-1,-.55,0,.55,1]):
            z=base+rise*abs(frac)**1.8
            m.vert((width*frac,y,z),(k/4,j/3))
    for j in range(len(sections)-1):
        for i in range(4):
            a=j*5+i+1
            m.quad(a,a+1,a+6,a+5)
    meshes.append(m)
    # Rolled top edges give a credible closed silhouette in oblique views.
    for side in [-1,1]:
        points=[(side*w,y,b+h) for y,b,w,h in sections]
        swept_tube('spout_lip_'+('left' if side<0 else 'right'),
                   'bright_steel',(.87,.88,.84),points,.0014,.0014,8)


def build_geometry():
    # Generic octagonal lower boiler (not a branded copy), tapering distinctly
    # through the shoulder; all dimensions remain close to the provisional pot.
    lathe('lower_chamber','aluminium',(.68,.73,.73),
          [(0,-.068),(.029,-.068),(.034,-.066),(.038,-.061),
           (.040,-.053),(.040,-.031),(.038,-.018),(.035,-.010),
           (.027,-.003),(.025,-.001)],sides=64,facet_strength=.95,facet_count=8)
    lathe('boiler_foot','dark_steel',(.40,.44,.44),
          [(.028,-.068),(.033,-.068),(.035,-.067),(.034,-.065)],sides=64,
          facet_strength=.9,facet_count=8)
    lathe('seam_gasket','gasket',(.14,.17,.18),
          [(.025,-.003),(.033,-.003),(.035,-.002),(.035,.001),
           (.027,.002)],sides=64)
    lathe('joint_band','bright_steel',(.86,.87,.84),
          [(.028,.001),(.033,.001),(.035,.002),(.035,.004),(.029,.005)],sides=64)
    # Upper collector widens toward the closed lid, its ten broad panels gentler
    # than the boiler facets; the thin waist and contrasting joint remain clear.
    lathe('upper_chamber','steel',(.79,.83,.83),
          [(0,.003),(.026,.003),(.029,.006),(.032,.011),
           (.034,.020),(.037,.041),(.038,.053),(.037,.060),
           (.034,.063)],sides=80,facet_strength=.56,facet_count=10)
    lathe('rim','bright_steel',(.87,.88,.84),
          [(.030,.061),(.037,.061),(.038,.063),(.037,.065),(.032,.066)],sides=64)
    # A closed, shallow domed lid and centered knob. The plate spans the mouth.
    lathe('closed_lid','steel',(.73,.77,.76),
          [(0,.068),(.022,.068),(.030,.067),(.035,.067),
           (.037,.069),(.036,.071),(.030,.072),(.018,.073),
           (0,.074)],sides=64)
    lathe('lid_knob','polymer',(.12,.14,.15),
          [(0,.074),(.006,.074),(.009,.076),(.009,.080),
           (.007,.083),(.003,.084),(0,.084)],sides=32)
    box('hinge_socket','dark_steel',(.30,.33,.33),(0,.036,.067),(.010,.005,.004))
    swept_tube('hinge_pin','bright_steel',(.80,.80,.77),
               [(-.009,.039,.069),(.009,.039,.069)],.002,.002,12)
    for suffix,z in [('upper',.049),('lower',-.043)]:
        box('grip_mount_'+suffix,'dark_steel',(.33,.37,.38),
            (0,.038,z),(.009,.007,.004))
    path=[(0,.043,.049),(0,.053,.053),(0,.063,.051),
          (0,.073,.043),(0,.079,.029),(0,.080,.009),
          (0,.080,-.011),(0,.076,-.030),(0,.067,-.043),
          (0,.056,-.049),(0,.044,-.044)]
    swept_tube('curved_grip','polymer',(.14,.16,.17),path,.009,.006)
    spout()
    lathe('pressure_valve','bright_steel',(.82,.83,.80),
          [(0,-.032),(.004,-.032),(.005,-.030),(.005,-.025),
           (.004,-.024),(0,-.024)],sides=24)
    for i,v in enumerate(meshes[-1].v):
        meshes[-1].v[i]=(v[0]+.038,v[1],v[2])


def make_textures():
    # Native 2K synthesis: independent scanline grain, 1-pixel machining lines,
    # broad satin variation, and restrained metallic wear. No source image used.
    n=2048
    y=np.arange(n,dtype=np.float32)[:,None]
    x=np.arange(n,dtype=np.float32)[None,:]
    coarse=RNG.normal(0,1,(n,1)).astype(np.float32)
    coarse=np.convolve(coarse[:,0],np.ones(39,dtype=np.float32)/39,mode='same')[:,None]
    micro=RNG.normal(0,1,(n,1)).astype(np.float32)
    grain=np.sin(y*.47 + 0.8*np.sin(x*.015))*.012 + micro*.009 + coarse*.040
    brush=np.sin(y*2.63)*.004 + np.sin(y*.058 + x*.002)*.012
    radial=np.abs((x-n/2)/(n/2))
    arr=np.empty((n,n,3),dtype=np.uint8)
    for k,base in enumerate((192,199,199)):
        value=base + (grain+brush)*255 + (1-radial)*3 + (k-1)*coarse*30
        arr[:,:,k]=np.clip(value,0,255).astype(np.uint8)
    Image.fromarray(arr,'RGB').save(OUT/'brushed_aluminium_albedo_2k.png',optimize=True)
    rough=np.clip(.29 + .035*np.sin(y*.47) + .025*coarse + .006*np.cos(x*.018),.18,.48)
    Image.fromarray(np.uint8(rough*255),'L').save(OUT/'brushed_aluminium_roughness_2k.png',optimize=True)
    # Grip gets its own non-upscaled 2K satin-polymer pore pattern.
    pore=RNG.normal(0,1,(n,n)).astype(np.float32)
    line=np.sin(y*.31+x*.08)*.012
    pv=np.clip(33+8*pore+250*line,14,67).astype(np.uint8)
    grip=np.stack((pv,pv+3,pv+5),axis=2)
    Image.fromarray(grip,'RGB').save(OUT/'satin_grip_albedo_2k.png',optimize=True)
    gr=np.clip(.64+.055*pore+.015*np.sin(y*.3),.45,.82)
    Image.fromarray(np.uint8(gr*255),'L').save(OUT/'satin_grip_roughness_2k.png',optimize=True)
    # A separately authored neutral secondary-object texture for the Apache
    # microwave mesh; optional until the simulator material mapping is reviewed.
    n=1024; yy=np.arange(n,dtype=np.float32)[:,None]
    base=np.clip(166+7*np.sin(yy*.115)+RNG.normal(0,3,(n,n)),125,203).astype(np.uint8)
    Image.fromarray(np.stack((base,base+2,base+3),axis=2),'RGB').save(
        OUT/'neutral_microwave_metal_1k.png',optimize=True)


def make_mtl():
    lines=[]
    texture={'aluminium':'brushed_aluminium_albedo_2k.png',
             'steel':'brushed_aluminium_albedo_2k.png',
             'bright_steel':'brushed_aluminium_albedo_2k.png',
             'dark_steel':'brushed_aluminium_albedo_2k.png',
             'polymer':'satin_grip_albedo_2k.png'}
    for name,color in [('aluminium',(.78,.80,.79)),('steel',(.83,.84,.83)),
                       ('bright_steel',(.91,.91,.89)),('dark_steel',(.31,.34,.35)),
                       ('gasket',(.17,.18,.19)),('polymer',(.13,.15,.16)),
                       ('shadow',(.09,.09,.10))]:
        lines.extend([f'newmtl {name}',f'Kd {color[0]} {color[1]} {color[2]}',
                      f'Ks {.35 if name != "polymer" else .08} {.35 if name != "polymer" else .08} {.35 if name != "polymer" else .08}',
                      f'Ns {80 if name != "polymer" else 14}'])
        if name in texture: lines.append(f'map_Kd {texture[name]}')
        lines.append('')
    (OUT/'moka_materials.mtl').write_text('\n'.join(lines))


def collision():
    # Close, simple convex-ish geom union; no surface-only mesh used for contact.
    geoms=[
      {'type':'cylinder','pos':[0,0,-.048],'size':[.039,.020]},
      {'type':'cylinder','pos':[0,0,-.012],'size':[.037,.018]},
      {'type':'cylinder','pos':[0,0,.022],'size':[.036,.027]},
      {'type':'cylinder','pos':[0,0,.054],'size':[.038,.011]},
      {'type':'cylinder','pos':[0,0,.071],'size':[.009,.011]},
      {'type':'box','pos':[0,-.045,.053],'size':[.012,.012,.012]},
      {'type':'box','pos':[0,.078,.002],'size':[.010,.010,.043]},
    ]
    # Explicit finite tilt check; rotation about Y does not change the grip reach.
    samples=[]
    for g in geoms:
        x,y,z=g['pos']; a,b=g['size'][:2]
        if g['type']=='cylinder':
            r,h=a,b
            samples += [(x+dx*r,y+dy*r,z+dz*h)
                        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)] for dz in [-1,1]]
        else:
            sx,sy,sz=g['size']
            samples += [(x+dx*sx,y+dy*sy,z+dz*sz)
                        for dx in [-1,1] for dy in [-1,1] for dz in [-1,1]]
    arr=np.asarray(samples)
    cavity_lo=np.array([-.1435,-.0865,.024]); cavity_hi=np.array([.0635,.0805,.168])
    center=np.array([-.04,-.014,.096])
    fit={}
    for angle in [-60,60]:
        a=math.radians(angle)
        rot=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        transformed=arr@rot.T+center
        fit[str(angle)]=bool(np.all(transformed.min(axis=0)>cavity_lo) and
                             np.all(transformed.max(axis=0)<cavity_hi))
    report={'units':'m','body_diameter_m':.080,'visual_height_m':.152,
            'collision_aabb_min_m':arr.min(axis=0).round(6).tolist(),
            'collision_aabb_max_m':arr.max(axis=0).round(6).tolist(),
            'collision_height_m':float(np.ptp(arr[:,2])),
            'cavity_height_m':float(cavity_hi[2]-cavity_lo[2]),
            'test_center_world_m':center.tolist(),'tilt_fit':fit,
            'grip_outer_y_m':.088,'panda_finger_span_note':'80 mm shell and narrow waist require side/tool grip qualification',
            'geoms':geoms}
    assert report['collision_height_m'] > report['cavity_height_m']
    assert all(fit.values())
    (OUT/'collision_proxy.json').write_text(json.dumps(report,indent=2)+'\n')
    xml=['<!-- Candidate contact geoms. Insert into a reviewed LIBERO object XML only after native validation. -->']
    for g in geoms:
        v=lambda seq:' '.join(f'{x:.6f}' for x in seq)
        xml.append(f'<geom type="{g["type"]}" pos="{v(g["pos"])}" size="{v(g["size"])}" density="100" friction="0.95 0.3 0.1" group="0"/>')
    (OUT/'collision_geoms.xmlfrag').write_text('\n'.join(xml)+'\n')
    return report


def preview():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(11,5.2),facecolor='#f5f3ee')
    for idx,(elev,azim,title) in enumerate([(17,-67,'Three-quarter view'),(13,30,'Opposite side / spout'),(5,-90,'Profile + dimensions')]):
        ax=fig.add_subplot(1,3,idx+1,projection='3d')
        ax.set_facecolor('#f5f3ee')
        light=np.array([.38,-.48,.79]); light/=np.linalg.norm(light)
        camera=np.array([math.cos(math.radians(azim))*math.cos(math.radians(elev)),
                         math.sin(math.radians(azim))*math.cos(math.radians(elev)),
                         math.sin(math.radians(elev))])
        halfway=(light+camera); halfway/=np.linalg.norm(halfway)
        for m in meshes:
            verts=np.asarray(m.v)
            if not len(verts): continue
            faces=[verts[np.asarray(f)-1] for f in m.f]
            cols=[]
            for f in faces:
                normal=np.cross(f[1]-f[0],f[2]-f[0])
                nn=np.linalg.norm(normal)
                if nn: normal/=nn
                diffuse=.44+.52*max(0,float(normal@light))
                # Reflection bands are preview lighting, not texture fabrication.
                stripe=.11*math.cos(3.6*math.atan2(normal[1],normal[0])+.5)
                specular=.30*max(0,float(normal@halfway))**16
                if m.material in ('polymer','gasket','shadow'):
                    stripe*=.15; specular*=.12
                value=np.asarray(m.base_color)*(diffuse+stripe)+specular
                cols.append(tuple(np.clip(value,0,1)))
            coll=Poly3DCollection(faces,linewidths=0,edgecolor='none',zsort='average')
            coll.set_facecolor(cols)
            ax.add_collection3d(coll)
        ax.set(xlim=(-.083,.083),ylim=(-.083,.083),zlim=(-.071,.085),
               title=title)
        ax.set_box_aspect((1,1,1.3))
        ax.view_init(elev=elev,azim=azim)
        ax.set_axis_off()
    fig.suptitle('Original unbranded moka prop · 152 mm tall · 80 mm faceted boiler',fontsize=13)
    fig.savefig(OUT/'cpu_geometry_preview.png',dpi=170,bbox_inches='tight')
    plt.close(fig)
    # Full-resolution texture crop, showing actual generated detail rather than
    # a flat color swatch.
    swatches=Image.new('RGB',(1200,650),(247,245,240))
    d=ImageDraw.Draw(swatches)
    for i,(file,title) in enumerate([
       ('brushed_aluminium_albedo_2k.png','2K brushed metal · 1:1 crop'),
       ('satin_grip_albedo_2k.png','2K satin grip · 1:1 crop'),
       ('neutral_microwave_metal_1k.png','1K neutral oven · 1:1 crop')]):
        src=Image.open(OUT/file).convert('RGB')
        crop=src.crop((200,250,560,610))
        swatches.paste(crop,(20+i*395,80))
        d.text((20+i*395,40),title,fill=(25,25,25))
    swatches.save(OUT/'cpu_texture_preview.png')


def hashes():
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='sha256.json')
    d={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'sha256.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    for old in OUT.glob('*.obj'):
        old.unlink()
    build_geometry()
    make_textures()
    make_mtl()
    for m in meshes: m.obj()
    report=collision()
    preview()
    from build_mjcf import build as build_mjcf
    build_mjcf()
    hashes()
    print(json.dumps({'meshes':len(meshes),'vertices':sum(len(m.v) for m in meshes),
                      'faces':sum(len(m.f) for m in meshes),
                      'collision_height_m':report['collision_height_m'],
                      'tilt_fit':report['tilt_fit']},indent=2))
