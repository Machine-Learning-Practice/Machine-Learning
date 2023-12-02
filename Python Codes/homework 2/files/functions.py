# Map X_train to polynomial

def polynomial_feature_mapping(feature1, feature2):
    """
    Feature mapping function to polynomial features    
    """
    feature1 = np.atleast_1d(feature1)
    feature2 = np.atleast_1d(feature2)
    degree = 6
    mapped_features = []
    
    for i in range(1, degree + 1):
        for j in range(i + 1):
            mapped_features.append((feature1**(i - j) * (feature2**j)))
    
    return np.stack(mapped_features, axis=1)




# Plot Data


def plot_data(features, labels, positive_label="Label 1", negative_label="Label 0"):
    positive_indices = labels == 1
    negative_indices = labels == 0
    
    # Plot examples
    plt.plot(features[positive_indices, 0], features[positive_indices, 1], 'k+', label=positive_label)
    plt.plot(features[negative_indices, 0], features[negative_indices, 1], 'yo', label=negative_label)



# Visualize Decision Boundary

def visualize_decision_boundary(weights, bias, features, labels):
    # Credit to dibgerge on Github for this plotting code
     
    plot_data(features[:, 0:2], labels)
    
    if features.shape[1] <= 2:
        plot_x = np.array([min(features[:, 0]), max(features[:, 0])])
        plot_y = (-1. / weights[1]) * (weights[0] * plot_x + bias)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(polynomial_feature_mapping(u[i], v[j]), weights) + bias)
        
        # It's important to transpose z before calling contour       
        z = z.T
        
        # Plot z = 0
        plt.contour(u, v, z, levels=[0.5], colors="g")
