import os.path
import pandas as pd
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.frame_params_list = []

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def save_frame_parameters(self):
        """Save all frame parameters to a single CSV file."""
        if not self.frame_params_list or not self.output_path:
            return
        df = pd.DataFrame(self.frame_params_list)
        csv_path = os.path.join(self.output_path, 'frame_parameters.csv')
        df.to_csv(csv_path, index=False)

    def on_render(self, time: float, frame_time: float):
        if self.frame >= self.frame_count:
            self.save_frame_parameters()
            raise RuntimeError("I AM DONE")

        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        material_diffuse = np.random.uniform(0.0, 255.0, 3) / 255.0
        material_shininess = np.random.uniform(3.0, 20.0)
        light_position = np.random.uniform(-20.0, 20.0, 3)

        model_translation = np.random.uniform(-20.0, 20.0, 3)
        model_matrix = Matrix44.from_translation(model_translation)

        camera_position = [5.0, 5.0, 15.0]
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat_offset = np.random.uniform(-2.0, 2.0, 3)
        lookat_target = model_translation + lookat_offset
        lookat = Matrix44.look_at(
            camera_position,
            lookat_target,
            (0.0, 1.0, 0.0),
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGBA', self.wnd.fbo.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(self.output_path, f'image_{self.frame:04}.png'))

            frame_params = {
                'frame': self.frame,
                'model_translation_x': model_translation[0],
                'model_translation_y': model_translation[1],
                'model_translation_z': model_translation[2],
                'material_diffuse_r': material_diffuse[0],
                'material_diffuse_g': material_diffuse[1],
                'material_diffuse_b': material_diffuse[2],
                'material_shininess': material_shininess,
                'light_position_x': light_position[0],
                'light_position_y': light_position[1],
                'light_position_z': light_position[2],
                'camera_position_x': camera_position[0],
                'camera_position_y': camera_position[1],
                'camera_position_z': camera_position[2],
                'lookat_target_x': lookat_target[0],
                'lookat_target_y': lookat_target[1],
                'lookat_target_z': lookat_target[2]
            }
            self.frame_params_list.append(frame_params)
            self.frame += 1