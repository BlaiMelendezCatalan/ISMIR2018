Computing statistics:
*********************

$ ipython

> import statistics as S
> import results as R
> import plots as P
> stats = {}
> authors = ['marolt', 'lidy', 'tsipas']
> thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
> for a in authors:
>     stats[a] = {}
>     for t in thresholds:
>         stats[a][t] = S.compute_statistics(ref_dir='../annotations/mapped_annotations', est_dir='../estimations/' + a + '/formatted_estimations/threshold_' + str(t), ncpus=<ncpus>)
> results = R.produce_results(stats)
> P.plot_PR_curves(results)
> P.plot_errors_by_class(results)
