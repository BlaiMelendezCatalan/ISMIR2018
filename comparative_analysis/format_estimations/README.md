# Format estimations:

cd format\_estimations/

(In Ipython)
```python
import format as F
d = F.get\_music\_segments('../estimations/<author>/raw\_estimations/', style=style*)
F.get_formatted_gt('../estimations/<author>/formatted_estimations/', '../audio/testing_split', d)
F.check_estimations('../estimations/<author>/formatted_estimations/', '../audio/testing_split')
```

\*style = 'detection' for Lidy and Marolt. style = 'segmentation' for Tsipas.
