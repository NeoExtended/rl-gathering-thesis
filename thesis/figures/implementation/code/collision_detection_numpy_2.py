def update_locations(action, freespace, locations):
    new_locations = locations + action_map[action]
    valid_locations = freespace.ravel()[
        (new_locations[:,1] + new_locations[:,0] * freespace.shape[1])
    ]
    new_locations = np.where(valid_locations[:, np.newaxis], new_locations, locations)
    return new_locations