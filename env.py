import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path("cartpole.xml")
        self.data = mujoco.MjData(self.model)

        # Define observation space
        high = np.array([
            self.model.stat.extent,  # Cart position (model extent gives a rough bound)
            np.finfo(np.float32).max, # Pole angle (can go beyond pi, so large value)
            np.finfo(np.float32).max, # Cart velocity
            np.finfo(np.float32).max  # Pole angular velocity
        ], dtype=np.float32)
        low = -high

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Define action space
        self.action_space = spaces.Discrete(2) # Two actions: push left or push right

        # Environment specific parameters
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360 # 12 degrees converted to radians
        self.force_magnitude = 10.0 # Force applied to the cart

        # Rendering setup
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None # For 'rgb_array' mode

    def _get_obs(self):
        """Returns the current observation based on MuJoCo state."""
        # Concatenate cart position, pole angle, cart velocity, pole angular velocity
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):

        # Reset MuJoCo simulation to initial state
        mujoco.mj_resetData(self.model, self.data)

        # Apply slight random initial perturbations to position and angle
        self.data.qpos[0] = self.np_random.uniform(low=-0.05, high=0.05) # Cart position
        self.data.qpos[1] = self.np_random.uniform(low=-0.05, high=0.05) # Pole angle
        self.data.qvel[:] = 0.0 # Velocities start at zero

        # Re-compute forward dynamics to update dependent quantities (e.g., velocities can influence forces)
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Map action to control input: 0 -> -force, 1 -> +force
        if action == 0:
            self.data.ctrl[0] = -self.force_magnitude
        else: # action == 1
            self.data.ctrl[0] = self.force_magnitude

        # Advance the simulation by one timestep
        mujoco.mj_step(self.model, self.data)

        # Get new observation
        observation = self._get_obs()
        
        truncated = False

        # Calculate reward: +1 for every step the pole is upright and cart is within bounds
        reward = 1.0

        # Check termination conditions
        cart_position = self.data.qpos[0]
        pole_angle = self.data.qpos[1]

        terminated = bool(
            cart_position < -self.x_threshold
            or cart_position > self.x_threshold
            or pole_angle < -self.theta_threshold_radians
            or pole_angle > self.theta_threshold_radians
        )

        if self.render_mode is not None:
            self._render_frame()

        info = {}
        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        """Renders the current state of the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # You might want to adjust the camera, e.g., to follow the cart
                # self.viewer.cam.trackbodyid = self.model.body('cart').id
                # self.viewer.cam.distance = self.model.stat.extent * 1.5
                # self.viewer.cam.lookat[0] = 0 # Center X
                # self.viewer.cam.lookat[1] = 0 # Center Y
                # self.viewer.cam.lookat[2] = 0.5 # Center Z (slightly above ground)

            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                # MuJoCo renderer for offscreen rendering
                self.renderer = mujoco.MjRenderContextOffscreen(self.model, 0)
                # Set a reasonable resolution for rendering
                self.renderer.width = 640
                self.renderer.height = 480

            # Render the frame and return the RGB array
            # The render method returns a numpy array (H, W, 3)
            return self.renderer.read_pixels(width=self.renderer.width, height=self.renderer.height,
                                             depth=False, segmentation=False)


    def render(self):
        """Public render method that dispatches to _render_frame if mode is rgb_array."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """Cleans up resources (viewer, renderer)."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            # MuJoCo offscreen context needs to be freed
            self.renderer.free()
            self.renderer = None



