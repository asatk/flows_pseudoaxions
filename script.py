from fedhex import DEFAULT_SEED
from fedhex.pretrain.generation import GridGaussGenerator, RepeatStrategy, DiagCov
from fedhex.utils import print_msg, LOG_INFO


nsamp = 1e3
ndim = 2
ngausx = 5
ngausy = 5
seed = DEFAULT_SEED
sigma_gaus = 0.025
config_path = "config.json"

cov_strat = RepeatStrategy(
    DiagCov(ndim=ndim, sigma=sigma_gaus))
g = GridGaussGenerator(
    nsamp=nsamp, cov_strat=cov_strat, ngausx=ngausx, ngausy=ngausy, seed=seed)

samples, labels = g.generate()
data, cond, whiten_data, whiten_cond = g.preproc(ret_whiten=True)

if g.save(config_path=config_path):
    print_msg("Saving Config: SUCCESS", level=LOG_INFO)
else:
    print_msg("Saving Config: FAILURE", level=LOG_INFO)
