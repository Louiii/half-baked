import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from tqdm import tqdm
from flax.training import train_state, checkpoints


def generate_image(points, xx, yy, sigma, npix):
    """Generates a single image from a point cloud."""
    img = np.zeros((npix, npix))
    for px, py in points:
        dist2 = (xx - px)**2 + (yy - py)**2
        img += np.exp(-dist2 / (2 * sigma**2))
    if img.max() > 0:
        img /= img.max()  # Normalize to [0, 1]
    return img

def create_dataset(npix, batch_size, size=10000):
    """A generator function that yields batches of data on-the-fly."""
    x = np.linspace(0, 1, npix, endpoint=False) + .5 / npix
    xx, yy = np.meshgrid(x, x)
    sigma = 0.01

    num_batches = size // batch_size
    for _ in range(num_batches):
        batch_images = []
        for _ in range(batch_size):
            points = np.random.uniform(0, 1, (200, 2))
            img = generate_image(points, xx, yy, sigma, npix)
            batch_images.append(img)
        # Yield a batch of JAX arrays with shape (B, H, W, C)
        yield jnp.array(np.stack(batch_images, axis=0)[..., np.newaxis])


class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # Explicitly use Kaiming Uniform initializer to match PyTorch's default
        kaiming_init = nn.initializers.kaiming_uniform()

        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME', kernel_init=kaiming_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME', kernel_init=kaiming_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME', kernel_init=kaiming_init)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.latent_dim, kernel_size=(1, 1), kernel_init=kaiming_init)(x)
        return x

class Decoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # Explicitly use Kaiming Uniform initializer to match PyTorch's default
        kaiming_init = nn.initializers.kaiming_uniform()

        x = nn.ConvTranspose(features=128, kernel_size=(1, 1), kernel_init=kaiming_init)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME', kernel_init=kaiming_init)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME', kernel_init=kaiming_init)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(4, 4), strides=(2, 2), padding='SAME', kernel_init=kaiming_init)(x)
        x = nn.sigmoid(x)
        return x


class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float

    @nn.compact
    def __call__(self, inputs):
        # inputs shape: (B, H, W, C), C == embedding_dim
        # Define the embedding table
        embeddings = self.param(
            'embedding',
            nn.initializers.uniform(scale=1/self.num_embeddings),
            (self.num_embeddings, self.embedding_dim)
        )

        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate L2 distances between inputs and embeddings
        distances = (
            jnp.sum(flat_input**2, axis=1, keepdims=True)
            - 2 * jnp.dot(flat_input, embeddings.T)
            + jnp.sum(embeddings.T**2, axis=0, keepdims=True)
        )

        # Find the closest embedding for each vector
        encoding_indices = jnp.argmin(distances, axis=1)
        quantized = embeddings[encoding_indices]

        # VQ Loss
        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - flat_input) ** 2)
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(flat_input)) ** 2)
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-Through Estimator
        quantized_st = flat_input + jax.lax.stop_gradient(quantized - flat_input)
        quantized_st = quantized_st.reshape(inputs.shape)

        # Perplexity
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)
        avg_probs = jnp.mean(encodings, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return quantized_st, vq_loss, perplexity, encoding_indices


class VQVAE(nn.Module):
    latent_dim: int
    num_embeddings: int
    commitment_cost: float

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.vq = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.latent_dim,
            commitment_cost=self.commitment_cost
        )
        self.decoder = Decoder(latent_dim=self.latent_dim)

    def __call__(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, _ = self.vq(z)
        recon = self.decoder(quantized)
        return recon, vq_loss, perplexity

    def encode_to_indices(self, x):
        z = self.encoder(x)
        _, _, _, indices = self.vq(z)
        return indices

    def decode_from_indices(self, indices):
        quantized = self.vq.variables['params']['embedding'][indices]
        return self.decoder(quantized)

# --- Training Setup ---

# Create a custom TrainState to bundle model parameters and optimizer state
class VQVAETrainState(train_state.TrainState):
    # You can add more attributes here if needed, e.g., batch stats for BatchNorm
    pass

@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, key, model):
    """A single JIT-compiled training step."""
    def loss_fn(params):
        recon, vq_loss, perplexity = model.apply({'params': params}, batch, rngs={'dropout': key})
        recon_loss = jnp.mean((recon - batch) ** 2) # Mean Squared Error
        loss = recon_loss + vq_loss
        return loss, (recon_loss, vq_loss, perplexity)

    (loss, (recon_loss, vq_loss, perplexity)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {
        'loss': loss,
        'recon_loss': recon_loss,
        'vq_loss': vq_loss,
        'perplexity': perplexity
    }
    return state, metrics

@partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, model):
    """A JIT-compiled evaluation step for generating reconstructions."""
    recon, _, _ = model.apply({'params': state.params}, batch)
    return recon


if __name__ == "__main__":
    # --- Hyperparameters ---
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    latent_dim = 32
    num_embeddings = 512
    commitment_cost = 0.25
    npix = 128

    # --- Define an absolute directory for checkpoints ---
    ckpt_dir = os.path.abspath('./vqvae_checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Initialization ---
    key = jax.random.PRNGKey(42)
    key, model_key, data_key = jax.random.split(key, 3)

    model = VQVAE(
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost
    )
    
    # Initialize model parameters
    dummy_input = jnp.ones((1, npix, npix, 1))
    params = model.init(model_key, dummy_input)['params']
    
    # Initialize optimizer and state
    optimizer = optax.adam(learning_rate=lr)
    state = VQVAETrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # --- Restore checkpoint if one exists ---
    # This will load the latest checkpoint and update the state object in place.
    # If no checkpoint exists, it returns the original state object.
    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    print(f"Resuming training from step: {int(state.step)}")

    # --- Training Loop ---
    print("Starting VQ-VAE training...")
    for epoch in range(num_epochs):
        # Create a new data generator for each epoch
        train_loader = create_dataset(npix, batch_size, size=10000)
        
        # Use tqdm for a progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=10000 // batch_size)
        
        epoch_metrics = { 'loss': [], 'recon_loss': [], 'vq_loss': [], 'perplexity': [] }

        for batch in pbar:
            key, train_key = jax.random.split(key)
            state, metrics = train_step(state, batch, train_key, model)
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                recon_loss=f"{metrics['recon_loss']:.4f}",
                perp=f"{metrics['perplexity']:.2f}"
            )

        # --- Save checkpoint at the end of the epoch ---
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=state,
            step=state.step,
            overwrite=False, # Set to True to save only the latest checkpoint
            keep=3          # Keep the 3 most recent checkpoints
        )

        # Print average metrics for the epoch
        avg_loss = np.mean([float(x) for x in epoch_metrics['loss']])
        avg_recon_loss = np.mean([float(x) for x in epoch_metrics['recon_loss']])
        avg_vq_loss = np.mean([float(x) for x in epoch_metrics['vq_loss']])
        avg_perp = np.mean([float(x) for x in epoch_metrics['perplexity']])

        print(f"Epoch {epoch + 1} Avg: Loss={avg_loss:.4f}, Recon Loss={avg_recon_loss:.4f}, VQ Loss={avg_vq_loss:.4f}, Perplexity={avg_perp:.4f}")

    # --- Animation Generation ---
    print("\n🎥 Generating animation...")
    os.makedirs('images_jax', exist_ok=True)

    num_points = 200
    num_frames = 100
    sigma = 0.01
    x_coords = np.linspace(0, 1, npix, endpoint=False) + 1 / (2 * npix)
    xx, yy = np.meshgrid(x_coords, x_coords)

    points = np.random.uniform(0, 1, (num_points, 2))
    velocities = np.random.uniform(-0.005, 0.005, (num_points, 2))

    orig_imgs = []
    recon_imgs = []

    for i in tqdm(range(num_frames), desc="Generating frames"):
        orig_img = generate_image(points, xx, yy, sigma, npix)
        orig_imgs.append(orig_img)
        plt.imsave(f'images_jax/orig_frame_{i:04d}.png', orig_img, cmap='gray')

        # Prepare input tensor: (1, H, W, 1)
        input_tensor = jnp.array(orig_img)[jnp.newaxis, ..., jnp.newaxis]
        recon = eval_step(state, input_tensor, model)
        
        # Convert JAX array back to NumPy for plotting
        recon_img = np.array(recon.squeeze())
        recon_imgs.append(recon_img)
        plt.imsave(f'images_jax/recon_frame_{i:04d}.png', recon_img, cmap='gray')

        # Update points with bounce
        points += velocities
        out_x = (points[:, 0] < 0) | (points[:, 0] > 1)
        velocities[out_x, 0] *= -1
        out_y = (points[:, 1] < 0) | (points[:, 1] > 1)
        velocities[out_y, 1] *= -1


    # Create and save side-by-side animation
    fig, (ax_orig, ax_recon, ax_both) = plt.subplots(1, 3, figsize=(15, 5))
    ax_orig.set_title('Original')
    ax_orig.axis('off')
    ax_recon.set_title('Reconstructed')
    ax_recon.axis('off')
    ax_both.set_title('Original (R) Reconstructed (G, B)')
    ax_both.axis('off')
    plt.tight_layout()

    def col_im(i):
        return np.stack([orig_imgs[i], recon_imgs[i], recon_imgs[i]], axis=-1)

    im_orig = ax_orig.imshow(orig_imgs[0], cmap='gray', animated=True)
    im_recon = ax_recon.imshow(recon_imgs[0], cmap='gray', animated=True)
    im_both = ax_both.imshow(col_im(0), cmap='gray', animated=True)

    def update(frame):
        im_orig.set_array(orig_imgs[frame])
        im_recon.set_array(recon_imgs[frame])
        im_both.set_array(col_im(frame))
        return [im_orig, im_recon, im_both]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    # ani.save('vqvae_animation_jax.gif', writer='pillow')
    ani.save('vqvae.mp4', writer='ffmpeg', fps=20)
    plt.close(fig)
