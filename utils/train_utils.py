
def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			if key in ["farl_iou", "farl_dice", "mean_iou", "mean_dice", "loss_l2", "loss_lpips"]:
				mean_vals["val_metrics/" + key] = mean_vals.setdefault("val_metrics/" + key, []) + [output[key]]
			else:
				mean_vals["val_losses/" + key] = mean_vals.setdefault("val_losses/" + key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals
