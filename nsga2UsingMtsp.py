import numpy as np
from scipy.spatial.distance import pdist, squareform
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt

# Define the number of salesmen
num_salesmen = 2
# Generate random cities and their coordinates
np.random.seed(0)
num_cities = 10
city_names = ['City{}'.format(i+1) for i in range(num_cities)]
cities = np.random.rand(num_cities, 2)

# Calculate the distance matrix between all pairs of cities
dist_matrix = squareform(pdist(cities, metric='euclidean'))

# Define the function to evaluate a route for mTSP
def evaluate_route(route):
    # Calculate the total distance for all salesmen's routes
    total_distance = 0 
    for i in range(num_cities-1):
        total_distance += dist_matrix[route[i], route[i+1]]
    total_distance += dist_matrix[route[-1], route[0]]  # return to the starting city
    
    return (total_distance,)

# Define the problem as a permutation of integers
creator.create('Fitness', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register('indices', np.random.permutation, num_cities)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate_route)

# Define the genetic operators for NSGA2
toolbox.register('mate', tools.cxOrdered)#cxSimulatedBinaryBounded,low=0, up=num_cities-1, eta=20.0 multi_crossover and cxOrdered:ordered crossover 
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.1)
toolbox.register('select', tools.selNSGA2)

# Set the parameters for NSGA2
population_size = 10
num_generations = 500
crossover_prob = 0.5
mutation_prob = 0.2

# Create the initial population
population = toolbox.population(n=population_size)

# Initialize the lists to store the best fitness values and the population at each generation
best_fitnesses = []
populations = []

# Run the NSGA2 algorithm
for i in range(num_generations):
    # Apply genetic operators to the population
    offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)

    # Evaluate the fitness of the offspring
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # Combine the parent and offspring populations
    combined_population = population + offspring

    # Apply the NSGA2 selection operator to the combined population
    population = toolbox.select(combined_population, k=population_size)

    # Extract the best solution from the final population
    best_solution = tools.selBest(population, k=1)[0]

    # Divide the best solution into chunks for each salesman
    
    chunk_size = num_cities // num_salesmen
    if num_cities % num_salesmen > 0:
       chunk_size += 1

    # Divide the best solution into chunks for each salesman
    routes = [best_solution[i:i+chunk_size] for i in range(0, len(best_solution), chunk_size)]

# Add each city in the route to the corresponding salesman's list of cities
    salesman_cities = [[] for _ in range(num_salesmen)] #This will create a list containing num_salesmen empty sublists, 
                                                        #which can be used to store the cities visited by each salesman.
    for r, route in enumerate(routes):
           for i in range(len(route)):
               city_idx = route[i]
               city_name = "City {}".format(city_idx) # Replace "City {}" with the actual name of the city
               salesman_cities[r].append(city_name)

# Calculate the total distance and number of cities visited for each salesman
total_dists = [0] * num_salesmen
num_visited = [0] * num_cities
for r, route in enumerate(routes):
    for i in range(len(route)-1):
        total_dists[r] += dist_matrix[route[i], route[i+1]]
        num_visited[route[i]] += 1
    total_dists[r] += dist_matrix[route[-1], route[0]]
    num_visited[route[-1]] += 1


# Plot the initial population with noise
initial_population_coords = np.zeros((num_cities*population_size, 2))
for i, ind in enumerate(population):
    coords = cities[ind]
    noise = np.random.normal(0, 1, size=2)
    coords += noise
    initial_population_coords[i*num_cities:(i+1)*num_cities] = coords
plt.figure(figsize=(5,5))
plt.plot(initial_population_coords[:,0], initial_population_coords[:,1], 'o', alpha=0.5, label='Initial population')
for i, name in enumerate(city_names):
    plt.text(cities[i,0], cities[i,1], name, fontsize=10)
plt.xlabel('Distance')
plt.ylabel('Tour Length')
plt.legend()
plt.show()


"""we use the visited_cities list to keep track of which cities have been visited, 
and skip over any cities that have already been visited by another salesman. 
We also modify the routes list to start and end with the depot city, 
and calculate the distance for each salesman's route
"""
# Find the index of the depot city
depot_idx = 0

# Determine the number of cities per salesman, rounding down
chunk_size = num_cities // num_salesmen

# Determine the number of remaining cities
num_remainder = num_cities % num_salesmen

# Initialize a list to keep track of which cities have been visited
visited_cities = [False] * num_cities

# Initialize the routes and total distances for each salesman
routes = []
total_dists = []

for r in range(num_salesmen):
    # Determine the start and end indices for this salesman's cities
    start_idx = r * chunk_size
    if r < num_remainder:
        start_idx += r
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx += num_remainder
        end_idx = start_idx + chunk_size

    # Initialize this salesman's route with the depot city
    route = [depot_idx]

    # Visit the cities in this salesman's range, skipping over any cities that have already been visited
    for i in range(start_idx, end_idx):
        if not visited_cities[i]:
            route.append(i)
            visited_cities[i] = True

    # Add the depot city to the end of this salesman's route
    route.append(depot_idx)

    # Get this salesman's distance
    dist = 0
    for i in range(len(route) - 1):
        dist += dist_matrix[route[i], route[i+1]]
    total_dists.append(dist)

    # Add this salesman's route to the list of routes
    routes.append(route)

# Print the results for each salesman
for r in range(num_salesmen):
    # Get this salesman's route and distance
    route = routes[r]
    dist = total_dists[r]

    # Print the results for this salesman
    print("Salesman {}: visited {} cities, traveled {:.2f} distance".format(r+1, len(route) - 2, dist))
    print("Visited cities:", [city_names[i] for i in route[1:-1]])

    # Plot the route for the salesman
    salesman_route_coords = cities[route]
    plt.plot(salesman_route_coords[:,0], salesman_route_coords[:,1], 'o-', label='Optimal Route for Salesman {}'.format(r+1))

# Plot the depot city
depot_coords = cities[depot_idx]
plt.plot(depot_coords[0], depot_coords[1], 's', markersize=10, label='Depot')

# Add city names as annotations
for i, name in enumerate(city_names):
    plt.text(cities[i,0], cities[i,1], name, fontsize=10)
plt.title("Optimal Route")
plt.xlabel('Distance')
plt.ylabel('Tour Length')
plt.legend()
plt.show()

"""----Start Code for Shortest Route from here----"""
# Find the shortest route among all salesmen
shortest_route = routes[0]
shortest_dist = total_dists[0]
for r in range(1, num_salesmen):
    if total_dists[r] < shortest_dist:
        shortest_route = routes[r]
        shortest_dist = total_dists[r]

# Print the shortest distance and visited cities for the salesma
print('\n\nPrint the shortest distance traveled by the salesmen:')
print("Salesman {}: visited {} cities, traveled {:.2f} distance".format(shortest_route[0] + 1, len(routes[shortest_route[0]]) - 2, shortest_dist))
print("Visited cities:", [city_names[i] for i in routes[shortest_route[0]][1:-1]] + [city_names[depot_idx]])

# Construct the optimal route with the shortest route
optimal_route = [depot_idx] + routes[shortest_route[0]][1:-1] + [depot_idx]

# Get the coordinates for the cities in the optimal route
optimal_route_coords = cities[optimal_route]

# Plot the optimal route
plt.plot(optimal_route_coords[:,0], optimal_route_coords[:,1], 'o-', label='Shortest Route')
plt.plot(cities[depot_idx,0], cities[depot_idx,1], 'rs', label='Depot City')

# Add city names as annotations
for i, name in enumerate(city_names):
    if i in optimal_route:
        plt.text(cities[i,0], cities[i,1], name, fontsize=9, color='hotpink')
    else:
        plt.text(cities[i,0], cities[i,1], name, fontsize=9)

# Set x and y limits based on the coordinates of the cities
x_min, x_max = cities[:,0].min(), cities[:,0].max()
y_min, y_max = cities[:,1].min(), cities[:,1].max()
x_margin, y_margin = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1 # 10% margin
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

# Add a legend and show the plot
plt.title("Shortest Route")
plt.xlabel('Distance')
plt.ylabel('Tour Length')
plt.legend()
plt.show()

"""----End Here----"""

























