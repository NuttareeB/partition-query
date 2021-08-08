import math
from shinglebased import ShingleBased


class Cosine(ShingleBased):

    def __init__(self, k):
        super().__init__(k)

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 1.0
        if len(s0) < self.get_k() or len(s1) < self.get_k():
            return 0.0
        profile0 = self.get_profile(s0)
        profile1 = self.get_profile(s1)
        return self.similarity_profiles(profile0, profile1)

    def similarity_profiles(self, profile0, profile1):
        norm_profile0 = self._norm(profile0)
        norm_profile1 = self._norm(profile1)
        if norm_profile0 == 0:
            norm_profile0 = 0.00001
        if norm_profile1 == 0:
            norm_profile1 = 0.00001
        return self._dot_product(profile0, profile1) / norm_profile0 * norm_profile1

    @ staticmethod
    def _dot_product(profile0, profile1):
        small = profile1
        large = profile0
        if len(profile0) < len(profile1):
            small = profile0
            large = profile1
        agg = 0.0
        for k, v in small.items():
            i = large.get(k)
            if not i:
                continue
            agg += 1.0 * v * i
        return agg

    @ staticmethod
    def _norm(profile):
        agg = 0.0
        for k, v in profile.items():
            agg += 1.0 * v * v
        return math.sqrt(agg)
