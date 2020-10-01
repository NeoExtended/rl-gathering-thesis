def update_locations(action, freespace, locations):
    new_locations = []
    delta = action_map[action]
    for loc in locations:
        if freespace[loc[0]+delta[0]][loc[1]+delta[1]]:
            new_locations.append([loc[0]+delta[0], loc[1]+delta[1]])
        else:
            new_locations.append(loc)
    return new_locations