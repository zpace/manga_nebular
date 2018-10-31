class AbundanceSet(object):
    def __init__(self, solar_Z, solar_logOH12):
        self.solar_Z = solar_Z
        self.solar_logOH12 = solar_logOH12

    def logZ_to_logOH12(self, logZ):
        return self.solar_logOH12 + logZ

    def Z_to_logOH12(self, Z):
        return np.log10(Z / self.solar_Z) + self.solar_logOH12

    def logOH12_to_Z(self, logOH12):
        return self.solar_Z * 10.**(logOH12 + self.solar_logOH12)
