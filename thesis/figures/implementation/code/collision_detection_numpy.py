def update_locations(action, freespace, locations):
    new_locations = locations + action_map[action]
    valid_locations = freespace[tuple(new_locations.T)]
    new_locations = np.where(valid_locations[:, np.newaxis], new_locations, locations)
    return new_locations