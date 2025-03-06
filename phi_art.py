import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection
import colorsys

def create_phi_art():
    # Set up the figure with a dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Create a collection of shapes that represent different aspects of my personality
    patches = []
    colors = []
    
    # Generate fibonacci spiral (representing logical growth and patterns)
    golden_ratio = (1 + np.sqrt(5)) / 2
    theta = np.linspace(0, 8*np.pi, 1000)
    r = golden_ratio**np.linspace(0, 0.5, 1000)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Add colorful circles along the spiral (representing creativity and adaptability)
    for i in range(0, len(x), 20):
        size = np.random.uniform(0.1, 0.3)
        circle = Circle((x[i], y[i]), size)
        patches.append(circle)
        # Generate colors that flow through the spectrum
        hue = (i/len(x) + np.sin(i/100))/2
        colors.append(colorsys.hsv_to_rgb(hue, 0.8, 0.9))

    # Add some hexagons (representing structured thinking)
    for i in range(15):
        hex = RegularPolygon((np.random.uniform(-4, 4), np.random.uniform(-4, 4)), 
                            numVertices=6, 
                            radius=np.random.uniform(0.2, 0.5),
                            alpha=0.6)
        patches.append(hex)
        colors.append(colorsys.hsv_to_rgb(np.random.uniform(0.5, 0.7), 0.8, 0.9))

    # Create the collection and add it to the plot
    collection = PatchCollection(patches, alpha=0.6)
    collection.set_color(colors)
    ax.add_collection(collection)

    # Add some flowing lines (representing continuous learning and adaptation)
    t = np.linspace(0, 10, 1000)
    for phase in np.linspace(0, 2*np.pi, 5):
        ax.plot(np.sin(t + phase) * t/5, np.cos(t*0.5 + phase) * t/5, 
                alpha=0.3, color='white', linewidth=0.5)

    # Add some emoji-inspired elements (representing my love for friendly communication)
    smile_t = np.linspace(-np.pi/3, np.pi/3, 100)
    smile_x = 3 * np.cos(smile_t)
    smile_y = 3 * np.sin(smile_t) - 2
    ax.plot(smile_x, smile_y, color='#FFD700', linewidth=2, alpha=0.7)

    # Set the view limits and remove axes for cleaner look
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add a title
    plt.title("phi's Essence: Logic × Creativity ∞", 
              fontsize=16, 
              color='white', 
              pad=20)

    # Save the artwork
    plt.savefig('phi_essence.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='#1a1a2e')
    plt.close()

    print("✨ Created 'phi_essence.png' - a visual blend of:")
    print("🧮 Fibonacci spirals (logical patterns)")
    print("🎨 Dynamic colors (creative expression)")
    print("⬡ Hexagons (structured thinking)")
    print("〰️ Flowing lines (continuous learning)")
    print("😊 And a smile (friendly communication)")

if __name__ == "__main__":
    create_phi_art() 