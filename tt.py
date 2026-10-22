#第一行：U：文件没有被跟踪，没有添加至暂存区
#第二行：A：
+ ： git add
- ： git quit

撤销：放弃暂存

## 第三次修改：验证git add撤销

no changes added to commit (use "git add" and/or "git commit -a")
alec@alecdeMac-mini testxx % git remote -v
origin  https://github.com/Alec-2026/Learning.git (fetch)
origin  https://github.com/Alec-2026/Learning.git (push)
alec@alecdeMac-mini testxx % git config
error: no action specified
alec@alecdeMac-mini testxx % git config
error: no action specified
alec@alecdeMac-mini testxx % git log

## 撤销提交：git reset --soft HEAD^1


## 解决conflict：git add .
远程提交：git push origin main
推送成功：git push origin main:main




##线上提交的部分，以版本信息为例
lec@alecdeMac-mini testxx % git log   
commit 4d74b94be64d89075ef5e8cc35f72da607adfa69 (HEAD -> main, origin/main, origin/HEAD)
Merge: c8c1747 f50c54f
Author: Alec Lee <ace.lby2023@gmail.com>
Date:   Thu Oct 22 11:36:58 2026 +0800

    Merge branch 'main' of https://github.com/Alec-2026/Learning

commit c8c1747b85bc6113b216af3188f9919e5f02f424
Author: Alec Lee <ace.lby2023@gmail.com>
Date:   Thu Oct 22 11:33:28 2026 +0800

    第二次提交

commit f50c54f29187ac78691c503e708f0214899174a7
Author: Alec Lee <ace.lby2023@gmail.com>
Date:   Thu Oct 22 10:49:06 2026 +0800

    Remove tracked macOS system files

commit 2cd1af7df6899a09d7d9c5203af55c2ffefb15d1
Author: Alec Lee <ace.lby2023@gmail.com>
Date:   Thu Oct 22 10:44:40 2026 +0800

    第二次提交

commit 8d3554418800e3e3ca2650a78650415cb0c39a9e (tag: 测试tag)
:


##第二次测试冲突解决代号597

import datetime
from pprint import pprint
import warnings
from IPython.display import display
import meridian_geox as geox
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
)

pd.set_option("display.max_columns", None)

meridian_geox_root = None


##本地第二次提交，验证conflict解决
def _get_time_series(
    data: pd.DataFrame,
    column: str,
    analysis_config: api.AnalysisConfig,
) -> TimeSeries:
  """Extracts pre-test and test time series for a given column."""
  pivoted_data = util.pivot_and_sort_data(data, column)
  if analysis_config.pretest_end_date is not None:
    pretest_data = pivoted_data[
        pivoted_data.index <= analysis_config.pretest_end_date
    ]
  else:
    pretest_data = pivoted_data[
        pivoted_data.index < analysis_config.analysis_start_date
    ]
  pretest = jnp.array(pretest_data.values)
  test_data = pivoted_data[
      (pivoted_data.index >= analysis_config.analysis_start_date)
      & (pivoted_data.index <= analysis_config.analysis_end_date)
  ]
  test = jnp.array(test_data.values)
  return TimeSeries(
      pretest=pretest,
      test=test,
      pretest_dates=pretest_data.index,
      test_dates=test_data.index,
  )


##第四次提交测试解决conflict
@dataclasses.dataclass
class TimeSeries:
  """Time series arrays for analysis."""

  pretest: jnp.ndarray
  test: jnp.ndarray
  pretest_dates: pd.Index
  test_dates: pd.Index

@dataclasses.dataclass
class TreatmentMask:
  """Treatment mask and corresponding geo names."""

  mask: jnp.ndarray
  geos: list[str]


###本地：第五次测试冲突解决
metrics = analysis_result.results[cell_id]

    # 1. Initialize local min/max for Y-axis scaling for the current cell.
    local_min_cf, local_max_cf = np.inf, -np.inf
    local_min_pw, local_max_pw = np.inf, -np.inf
    local_min_l, local_max_l = np.inf, -np.inf
    local_min_i, local_max_i = np.inf, -np.inf

#####远程：第五次测试冲突解决
 # 2. Collect all finite values to determine local y-limits.
    # Observed vs. Counterfactual
    df_cf = metrics.counterfactual_conversions
    df_cf_test = df_cf[df_cf.index >= start_dt].copy()
    low_cf = df_cf_test['lower_bound']
    high_cf = df_cf_test['upper_bound']
    valid_obs = df_cf['observed'][np.isfinite(df_cf['observed'])]
    valid_cf = df_cf['counterfactual'][np.isfinite(df_cf['counterfactual'])]
    valid_low_cf = low_cf[np.isfinite(low_cf)]
    valid_high_cf = high_cf[np.isfinite(high_cf)]
    all_finite_cf = pd.concat(
        [valid_obs, valid_cf, valid_low_cf, valid_high_cf]
    )
    if not all_finite_cf.empty:
      local_min_cf = all_finite_cf.min()
      local_max_cf = all_finite_cf.max()

##第6.1次保留部分代码
 # Step 1: LOO Cross-Validation and Gap Calculation
  for k in range(num_dates):
    # Omit evaluation index k from the training dataset 
    train_x = np.delete(t, k)
    train_y = np.delete(y, k)

    # Fit OLS line: y_t ~ alpha + beta * t，公式为：
    # y_t = alpha + beta * t
    # y_t = y_t - alpha
    # y_t = y_t - beta * t
    slope, intercept = np.polyfit(train_x, train_y, 1)

    # Predict the omitted index k
    y_pred = intercept + slope * t[k]
    gaps.append(abs(y[k] - y_pred))

  gaps = np.array(gaps)

  # Step 2: Calculate the empirical IQR threshold
  q75, q25 = np.percentile(gaps, [75, 25])
  iqr = q75 - q25
  cutoff = q75 + (multiplier * iqr)

  return gaps, cutoff


##同步功能：测试
#再提交一个版本：第五行提交git commit -m "测试"
#拉取和推送同步：先pull再push
#在点击同步时可能会有冲突，需要手动解决冲突；最后再同步
#同步完成后，本地和远程的代码就会一致了


##9/新建分支：测试分支
#接下来的开发在branch_test1分支上进行

