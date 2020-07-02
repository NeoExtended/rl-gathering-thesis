def check_no_loop(locations, freespace):
    return freespace[tuple(locations.T)]