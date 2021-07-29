from torch import distributed as dist



def get_dist_info():
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    return rank, word_size