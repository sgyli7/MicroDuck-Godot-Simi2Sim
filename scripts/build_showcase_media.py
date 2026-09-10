"""Edit real-time capture files into a chaptered PV and compact README GIFs.

Requires FFmpeg with libass, libx264, palettegen and paletteuse. No synthesized
motion, speed changes or replacement robot frames are used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/media'
EDIT = ROOT / 'results/showcase/edit'
SKILLS = [
 ('standing','小小维修站','SERVICE BAY 01','MicroDuck · 九项策略引擎实录'),
 ('walking','出发，巡视工位','WALK / TURN / STOP','WD05 · 前进、转向与停步'),
 ('sitstand','收起身形，再站起来','SIT / STAND','Sitstand_Godot · 坐下与起身'),
 ('ground_pick','低头，贴近地面','GROUND PICK','alpha_ground_pick · 低头拾取动作'),
 ('kick_left','左脚，精准触球','LEFT KICK','K10 · 左脚触球后保持站立'),
 ('kick_right','换一只脚','RIGHT KICK','KR06 · 右脚触球与姿态恢复'),
 ('roulade','翻过去，站回来','FORWARD ROLL','R11 · 前滚一周后自主恢复站立'),
 ('roller','轮足：推进、滑行、刹车','ROLLER / WORK IN PROGRESS','原版 roller · 刹车未达标，仍有反向滑动'),
 ('roller_crouch','压低重心，滑出工位','CROUCH / GLIDE / RISE','原版 roller_crouch · 下蹲滑行后起身'),
]

def run(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def ass(path, title, subtitle, number, duration):
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Text,Noto Sans CJK SC,48,&H00E8E5DD,&H00E8E5DD,&H00252324,&H00252324,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    def line(text, x, y, size, color='E8E5DD', extra=''):
        return 'Dialogue: 0,0:00:00.00,0:02:00.00,Text,,0,0,0,,{\\pos(%d,%d)\\fs%d\\1c&H%s&%s}%s\n' % (x,y,size,color,extra,text)
    content = line('MICRODUCK',64,22,32,'38BDE0',r'\b1\fsp5')
    content += line('SERVICE BAY 01   /   小小维修站',374,28,24)
    content += line(f'{number:02d} / 09',1750,22,32,'9E7289')
    content += line(title,64,938,54,extra=r'\fad(160,0)')
    content += line(subtitle,67,1011,28,'BABEBE')
    content += line('ONNX × GODOT / 实时策略控制',1370,1020,24,'38BDE0')
    path.write_text(header + content)


def build(prefix):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('FFmpeg is required')
    MEDIA.mkdir(parents=True,exist_ok=True); EDIT.mkdir(parents=True,exist_ok=True)
    segments=[]; recipe=[]; offset=0.
    common=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2','-filter_threads','2']
    for i,(skill,title,english,subtitle) in enumerate(SKILLS,1):
        directory=ROOT/'results/showcase'/f'{prefix}_{skill}'
        data=json.loads((directory/'capture.json').read_text())
        # The full action duration is retained. The standing introduction uses
        # three seconds; its full ten-second recording remains in the archive.
        first=data['entry_seconds']; last=first+data['skill_seconds']
        if skill=='standing':first=3.;last=6.
        frames=data['frames'];base=frames[0]['milliseconds']
        start=min(frames,key=lambda f:abs(f['sim_seconds']-first))['milliseconds']
        finish=min(frames,key=lambda f:abs(f['sim_seconds']-last))['milliseconds']
        start=(start-base)/1000.;duration=(finish-base)/1000.-start
        subtitles=EDIT/f'{skill}.ass';ass(subtitles,title,subtitle,i,duration)
        output=EDIT/f'{skill}.mp4'
        filters=(f'fps=30,drawbox=x=0:y=0:w=iw:h=82:color=0x24232b:t=fill,'
                 f'drawbox=x=0:y=925:w=iw:h=155:color=0x24232b:t=fill,'
                 f'drawbox=x=64:y=919:w=130:h=6:color=0xe0bd38:t=fill,'
                 f"ass=filename='{subtitles}':fontsdir='{ROOT / 'godot/atelier/fonts'}',"
                 f'fade=t=in:st=0:d=0.13,fade=t=out:st={max(0,duration-.13):.4f}:d=0.13')
        run(common+['-ss',f'{start:.6f}','-i',str(directory/'raw.mp4'),'-t',f'{duration:.6f}',
                    '-vf',filters,'-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(output)])
        segments.append(output)
        recipe.append({'skill':skill,'source':str((directory/'raw.mp4').relative_to(ROOT)),
                       'source_sha256':hashlib.sha256((directory/'raw.mp4').read_bytes()).hexdigest(),
                       'source_start_seconds':start,'duration_seconds':duration,'pv_start_seconds':offset,
                       'speed':1.0,'chapter':english})
        offset+=round(duration*30)/30
        print('Edited',skill,flush=True)
    concat=EDIT/'chapters.txt';concat.write_text(''.join(f"file '{p}'\n" for p in segments))
    run(common+['-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(MEDIA/'microduck-service-bay-pv.mp4')])
    # GIF edit points refer to already captioned chapters and never accelerate
    # motion. Cuts compress idle holds; roll and crouch recovery remain intact.
    excerpts={'standing':[(0,2.5)],'walking':[(2.5,6.8)],'sitstand':[(0,1.5),(5.8,7.8)],
              'ground_pick':[(0,3.9)],'kick_left':[(0,2.4)],'kick_right':[(0,2.4)],
              'roulade':[(0,4.9)],'roller':[(.1,1.6),(6.5,9.8)],'roller_crouch':[(0,4.9)]}
    cuts=[]
    for segment,(skill,*_) in zip(segments,SKILLS):
        for k,(start,end) in enumerate(excerpts[skill]):
            output=EDIT/f'preview_{skill}_{k}.mp4'
            run(common+['-ss',str(start),'-i',str(segment),'-t',str(end-start),'-an','-c:v','libx264',
                        '-threads','2','-preset','fast','-crf','20','-vf','scale=768:432',str(output)])
            cuts.append(output)
    short=EDIT/'preview.txt';short.write_text(''.join(f"file '{p}'\n" for p in cuts))
    run(common+['-f','concat','-safe','0','-i',str(short),'-c','copy',str(EDIT/'preview.mp4')])
    gif(common,EDIT/'preview.mp4',MEDIA/'microduck-service-bay.gif',640,12)
    for filename,names in [('postures',['sitstand','ground_pick']),('kicks-and-roll',['kick_left','kick_right','roulade']),('wheels',['roller','roller_crouch'])]:
        path=EDIT/f'{filename}.txt';path.write_text(''.join(f"file '{EDIT / (name+'.mp4')}'\n" for name in names))
        video=EDIT/f'{filename}.mp4'
        run(common+['-f','concat','-safe','0','-i',str(path),'-c','copy',str(video)])
        gif(common,video,MEDIA/f'{filename}.gif',640,12)
    run(common+['-ss','3.3','-i',str(EDIT/'walking.mp4'),'-frames:v','1',str(MEDIA/'poster.jpg')])
    (MEDIA/'edit.json').write_text(json.dumps({'source_prefix':prefix,'chapters':recipe,
        'gif_excerpts':excerpts,'fps_mp4':30,'gif_fps_target':12,'gif_leading_trim_seconds':.2,'silent':True,
        'note':'Real captured frames at wall-clock speed. Cuts, letterbox graphics and captions only. GIF cuts shorten holds; MP4 retains full action durations except standing introduction.'},ensure_ascii=False,indent=2)+'\n')


def gif(common,source,dest,width,fps):
    # Flat comic colors compress more cleanly without moving dither noise.
    hero = dest.name == 'microduck-service-bay.gif'
    budget = 15_000_000 if hero else 10_000_000
    profiles = [(640,10,64),(640,10,48)] if hero else [(width,fps,96),(560,10,64),(480,10,48)]
    for size,rate,colors in profiles:
        filters=(f'fps={rate},scale={size}:-2:flags=lanczos,split[a][b];'
                 f'[a]palettegen=max_colors={colors}:stats_mode=diff[p];'
                 '[b][p]paletteuse=dither=none:diff_mode=rectangle')
        run(common+['-ss','0.20','-i',str(source),'-filter_complex_threads','2',
                    '-filter_complex',filters,'-loop','0',str(dest)])
        if dest.stat().st_size<=budget:
            break
    if dest.stat().st_size>budget:
        raise RuntimeError(f'GIF exceeds homepage size budget: {dest}')
    print(dest.name,dest.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='04')
    build(parser.parse_args().prefix)
