
# keep track of train loss over validation loss ration
# keep training model until the ratio decrease to a certain ratio
def cond_learn(learner, lr, threshold = 1.01, is_cycle = False, cycle_l = 2, is_enlarge_c = False):
    mult = 1
    i = 0
    t_loss = 1.5
    v_loss = 1
    # keep going if training loss bigger than validation loss over threshold
    while t_loss / v_loss > threshold:
        i += 1
        # enable cosine annealing
        if is_cycle:
            _, ep_vals = learn.fit(lr, 1, cycle_len = cycle_l, get_ep_vals = True)
        # enable enlarging cosine annealing
        elif is_enlarge_c:
            _, ep_vals = learn.fit(lr, 1, cycle_len = mult)
            mult += 1
        else:
            _, ep_vals = learn.fit(lr, 1, get_ep_vals = True)
        t_loss = list(ep_vals.values())[0][0]
        v_loss = list(ep_vals.values())[0][1]
        acc = list(ep_vals.values())[0][2]
    return learner

# calculate accuarcy
def accuracy_custom(preds, tags):
    preds_tags = np.argmax(preds, 1)
    n = len(tags)
    return sum(preds_tags == tags)/n

# Visualize trn loss, val loss and val accuracy
def vis_loss_acc(learn, figsize = (8, 12)):
    n_epoch = learn.sched.epoch
    bs = int(len(learn.sched.losses) / n_epoch)
    # validation loss
    val_loss = learn.sched.val_losses
    trn_loss = [learn.sched.losses[i] for i in range(bs-1, n_epoch*bs, bs)]
    # validation accuracy
    val_acc = learn.sched.rec_metrics
    # visualization
    fig, ax = plt.subplots(2, 1, figsize = figsize)
    iters = list(range(n_epoch))
    ax[0].plot(iters, val_loss, label = 'Validation Loss')
    ax[0].plot(iters, trn_loss, label = 'Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc = 'upper right')
    ax[1].plot(iters, val_acc)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy');

def fetch_data(sz, bs = 64, cpu_workers = 60):
    tfms = tfms_from_model(arch, sz, aug_tfms = transforms_side_on, max_zoom = 1.1)
    data = ImageClassifierData.from_paths(PATH,
                                          tfms = tfms,
                                          trn_name = 'imgtrain',
                                          val_name = 'imgval',
                                          test_name = 'imgtest',
                                          test_with_labels = True,
                                          num_workers = cpu_workers,
                                          bs = bs)
    return data if sz>300 else data.resize(340, 'tmp_30class')
