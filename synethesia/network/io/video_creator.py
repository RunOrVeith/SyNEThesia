#! /usr/bin/env python3

import ffmpy
from pathlib import Path


class VideoCreator(object):

    def __init__(self, fps=24, resolution=(1920, 1080), delete_frames=True):
        self.fps = fps
        self.resolution = resolution
        self.delete_frames = delete_frames

    def __call__(self, png_folder, mp3_file, output_name="music_video.mp4"):
        png_folder = Path(png_folder).resolve()
        mp3_file = Path(mp3_file).resolve()
        if not png_folder.is_dir():
            raise ValueError(f"{str(png_folder)} is not a directory.")
        if not mp3_file.is_file():
            raise ValueError(f"{str(mp3_file)} is not a valid file.")

        intermediate_name = "video.mp4"
        video_no_audio = self.merge_frames(png_folder=png_folder, output_name=intermediate_name)
        video_audio = self.add_sound_to_video(video_file=video_no_audio, sound_file=mp3_file,
                                              target_file=png_folder / output_name)
        self.maybe_delete_pngs(png_folder=png_folder)
        video_no_audio.unlink()
        return video_audio

    def maybe_delete_pngs(self, png_folder):
        if not self.delete_frames:
            return

        for pth in png_folder.iterdir():
            if pth.is_file() and pth.suffix == ".png":
                pth.unlink()

    def merge_frames(self, png_folder, output_name):
        resolution = 'x'.join(map(str, self.resolution))
        output = png_folder / output_name
        concatenate = ffmpy.FFmpeg(inputs={str(png_folder / "*.png"):
                                          f"-r {self.fps} -f image2 -s {resolution} -pattern_type glob"},
                                   outputs={str(output):
                                            f"-y -vcodec libx264 -crf {self.fps}"})
        concatenate.run()
        assert output.is_file()
        return output


    def add_sound_to_video(self, video_file, sound_file, target_file):

        merge = ffmpy.FFmpeg(inputs={str(video_file): None, str(sound_file): None},
                             outputs={str(target_file): "-y -c:v copy -c:a aac -strict experimental -shortest"})
        merge.run()
        assert target_file.is_file()
        return target_file
