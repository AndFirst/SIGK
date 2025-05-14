import math
import os.path

import moderngl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from base_window import BaseWindow
from PIL import Image
from pyrr import Matrix44


class ImageGenerator(nn.Module):
    def __init__(self, input_dim, output_size):
        super(ImageGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 4 * 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.fc(x)
        features = features.view(batch_size, 128, 4, 4)
        return self.decoder(features)


class PhongWindow(BaseWindow):
    def __init__(
        self,
        renderer_type="normal",
        model_path=None,
        input_dim=24,
        output_size=128,
        **kwargs,
    ):
        self.renderer_type = "neural"
        self.model_path = "../notebook/checkpoints/best_model.pth"
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.frame_params_list = []
        self.output_size = output_size
        if self.renderer_type == "neural":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ImageGenerator(
                input_dim=input_dim, output_size=output_size
            ).to(self.device)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            self.camera_pos = [5.0, 5.0, 15.0]
            self.max_r = math.sqrt(25**2 + 25**2 + 35**2)
            self.r_range = [0, self.max_r]
            self.theta_range = [0, math.pi]
            self.phi_range = [-math.pi, math.pi]
            self.shininess_range = [3, 20]
            self.diffuse_range = [0, 1]
            self.init_neural_renderer()

    def init_neural_renderer(self):
        self.neural_program = self.ctx.program(
            vertex_shader="""
                
                in vec2 in_vert;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() {
                    f_color = texture(texture0, v_texcoord);
                }
            """,
        )
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        indices = np.array([0, 1, 2, 2, 1, 3], dtype="i4")
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        self.neural_vao = self.ctx.vertex_array(
            self.neural_program, [(vbo, "2f 2f", "in_vert", "in_texcoord")], ibo
        )
        self.texture = self.ctx.texture((self.output_size, self.output_size), 3)
        self.texture.use(0)

    def init_shaders_variables(self):
        if self.renderer_type == "normal":
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
        csv_path = os.path.join(self.output_path, "frame_parameters.csv")
        df.to_csv(csv_path, index=False)

    def normalize(self, value, range_min, range_max):
        return 2 * ((value - range_min) / (range_max - range_min)) - 1

    def get_orthonormal_basis(self, z_prime):
        ref = [0, 0, 1] if abs(z_prime[2]) <= 0.99 else [0, 1, 0]
        x_prime = [
            z_prime[1] * ref[2] - z_prime[2] * ref[1],
            z_prime[2] * ref[0] - z_prime[0] * ref[2],
            z_prime[0] * ref[1] - z_prime[1] * ref[0],
        ]
        x_norm = math.sqrt(sum(x * x for x in x_prime))
        x_prime = [x / x_norm for x in x_prime]
        y_prime = [
            z_prime[1] * x_prime[2] - z_prime[2] * x_prime[1],
            z_prime[2] * x_prime[0] - z_prime[0] * x_prime[2],
            z_prime[0] * x_prime[1] - z_prime[1] * x_prime[0],
        ]
        y_norm = math.sqrt(sum(y * y for y in y_prime))
        y_prime = [y / y_norm for y in y_prime]
        return x_prime, y_prime, z_prime

    def cartesian_to_spherical_lookat(self, x, y, z, lookat_vec):
        x_prime, y_prime, z_prime = self.get_orthonormal_basis(lookat_vec)
        x_new = x * x_prime[0] + y * x_prime[1] + z * x_prime[2]
        y_new = x * y_prime[0] + y * y_prime[1] + z * y_prime[2]
        z_new = x * z_prime[0] + y * z_prime[1] + z * z_prime[2]
        r = math.sqrt(x_new**2 + y_new**2 + z_new**2)
        theta = math.acos(z_new / r) if r > 0 else 0
        phi = math.atan2(y_new, x_new)
        return r, theta, phi

    def preprocess_neural_params(
        self,
        model_translation,
        material_diffuse,
        material_shininess,
        light_position,
        lookat_target,
    ):
        """Preprocess parameters for neural renderer, mirroring FrameDataset logic."""
        lookat_x_rel = lookat_target[0] - self.camera_pos[0]
        lookat_y_rel = lookat_target[1] - self.camera_pos[1]
        lookat_z_rel = lookat_target[2] - self.camera_pos[2]
        lookat_norm = math.sqrt(lookat_x_rel**2 + lookat_y_rel**2 + lookat_z_rel**2)
        lookat_vec = (
            [
                lookat_x_rel / lookat_norm,
                lookat_y_rel / lookat_norm,
                lookat_z_rel / lookat_norm,
            ]
            if lookat_norm > 0
            else [0, 0, 0]
        )
        model_x_rel = model_translation[0] - self.camera_pos[0]
        model_y_rel = model_translation[1] - self.camera_pos[1]
        model_z_rel = model_translation[2] - self.camera_pos[2]
        model_r, model_theta, model_phi = self.cartesian_to_spherical_lookat(
            model_x_rel, model_y_rel, model_z_rel, lookat_vec
        )
        light_x_rel = light_position[0] - self.camera_pos[0]
        light_y_rel = light_position[1] - self.camera_pos[1]
        light_z_rel = light_position[2] - self.camera_pos[2]
        light_r, light_theta, light_phi = self.cartesian_to_spherical_lookat(
            light_x_rel, light_y_rel, light_z_rel, lookat_vec
        )
        light_to_model_vec = [
            light_x_rel - model_x_rel,
            light_y_rel - model_y_rel,
            light_z_rel - model_z_rel,
        ]
        ltm_norm = math.sqrt(sum(x**2 for x in light_to_model_vec))
        light_to_model_unit = [
            x / ltm_norm if ltm_norm > 0 else 0 for x in light_to_model_vec
        ]
        cos_view_light = sum(l * v for l, v in zip(light_to_model_unit, lookat_vec))
        ltm_normalized = self.normalize(ltm_norm, 0, self.max_r)
        x_prime, y_prime, z_prime = self.get_orthonormal_basis(lookat_vec)

        def rotate_to_view(x, y, z):
            return [
                x * x_prime[0] + y * x_prime[1] + z * x_prime[2],
                x * y_prime[0] + y * y_prime[1] + z * y_prime[2],
                x * z_prime[0] + y * z_prime[1] + z * z_prime[2],
            ]

        model_view_xyz = rotate_to_view(model_x_rel, model_y_rel, model_z_rel)
        light_view_xyz = rotate_to_view(light_x_rel, light_y_rel, light_z_rel)
        params = [
            self.normalize(model_r, *self.r_range),
            self.normalize(model_theta, *self.theta_range),
            self.normalize(model_phi, *self.phi_range),
            self.normalize(light_r, *self.r_range),
            self.normalize(light_theta, *self.theta_range),
            self.normalize(light_phi, *self.phi_range),
            *lookat_vec,
            self.normalize(material_diffuse[0], *self.diffuse_range),
            self.normalize(material_diffuse[1], *self.diffuse_range),
            self.normalize(material_diffuse[2], *self.diffuse_range),
            self.normalize(material_shininess, *self.shininess_range),
            ltm_normalized,
            cos_view_light,
            *light_to_model_unit,
            *model_view_xyz,
            *light_view_xyz,
        ]
        return torch.tensor(params, dtype=torch.float32).to(self.device)

    def on_render(self, time: float, frame_time: float):
        if self.frame >= self.frame_count:
            self.save_frame_parameters()
            raise RuntimeError("I AM DONE")
        material_diffuse = np.random.uniform(0.0, 1.0, 3)
        material_shininess = np.random.uniform(3.0, 20.0)
        light_position = np.random.uniform(-20.0, 20.0, 3)
        model_translation = np.random.uniform(-20.0, 20.0, 3)
        camera_position = [5.0, 5.0, 15.0]
        lookat_offset = np.random.uniform(-2.0, 2.0, 3)
        lookat_target = model_translation + lookat_offset
        print(self.renderer_type)
        if self.renderer_type == "normal":
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            model_matrix = Matrix44.from_translation(model_translation)
            proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
            lookat = Matrix44.look_at(
                camera_position,
                lookat_target,
                (0.0, 1.0, 0.0),
            )
            model_view_projection = proj * lookat * model_matrix
            self.model_view_projection.write(
                model_view_projection.astype("f4").tobytes()
            )
            self.model_matrix.write(model_matrix.astype("f4").tobytes())
            self.material_diffuse.write(
                np.array(material_diffuse, dtype="f4").tobytes()
            )
            self.material_shininess.write(
                np.array([material_shininess], dtype="f4").tobytes()
            )
            self.light_position.write(np.array(light_position, dtype="f4").tobytes())
            self.camera_position.write(np.array(camera_position, dtype="f4").tobytes())
            self.vao.render()
            if self.output_path:
                img = Image.frombuffer(
                    "RGBA", self.wnd.fbo.size, self.wnd.fbo.read(components=4)
                ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                img = img.convert("RGB")
                img.save(os.path.join(self.output_path, f"image_{self.frame:04}.png"))
        elif self.renderer_type == "neural":
            params = self.preprocess_neural_params(
                model_translation,
                material_diffuse,
                material_shininess,
                light_position,
                lookat_target,
            )
            with torch.no_grad():
                input_params = params.unsqueeze(0)
                output = self.model(input_params)
                output = (output + 1) / 2
                output = output.clamp(0, 1)
                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray((output * 255).astype(np.uint8), "RGB")
                if self.output_path:
                    img.save(
                        os.path.join(self.output_path, f"image_{self.frame:04}.png")
                    )
                texture_data = (output * 255).astype(np.uint8).tobytes()
                self.texture.write(texture_data)
                self.ctx.clear(0.0, 0.0, 0.0, 0.0)
                self.ctx.enable(moderngl.BLEND)
                self.neural_program["texture0"].value = 0
                self.neural_vao.render(moderngl.TRIANGLES)
                self.ctx.disable(moderngl.BLEND)
        else:
            raise ValueError(f"Unknown renderer_type: {self.renderer_type}")
        if self.output_path:
            frame_params = {
                "frame": self.frame,
                "model_translation_x": model_translation[0],
                "model_translation_y": model_translation[1],
                "model_translation_z": model_translation[2],
                "material_diffuse_r": material_diffuse[0],
                "material_diffuse_g": material_diffuse[1],
                "material_diffuse_b": material_diffuse[2],
                "material_shininess": material_shininess,
                "light_position_x": light_position[0],
                "light_position_y": light_position[1],
                "light_position_z": light_position[2],
                "camera_position_x": camera_position[0],
                "camera_position_y": camera_position[1],
                "camera_position_z": camera_position[2],
                "lookat_target_x": lookat_target[0],
                "lookat_target_y": lookat_target[1],
                "lookat_target_z": lookat_target[2],
            }
            self.frame_params_list.append(frame_params)
            self.frame += 1
