import os
import _pickle as pkl


def create_log_dirs(args, modelparam, get_dir_names_only=False):
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    text_logs_dir = os.path.join(args.log_dir, "text_logs")
    ckpt_logs_dir = os.path.join(args.log_dir, "ckpt")
    tb_logs_dir = os.path.join(args.log_dir, "tb")

    if not os.path.isdir(text_logs_dir):
        os.makedirs(text_logs_dir)
    if not os.path.isdir(ckpt_logs_dir):
        os.makedirs(ckpt_logs_dir)
    if not os.path.isdir(tb_logs_dir):
        os.makedirs(tb_logs_dir)

    text_logs_dir_mp = os.path.join(text_logs_dir, modelparam)
    if not os.path.isdir(text_logs_dir_mp):
        os.makedirs(text_logs_dir_mp)

    ckpt_logs_dir_mp = os.path.join(ckpt_logs_dir, modelparam)
    if not os.path.isdir(ckpt_logs_dir_mp):
        os.makedirs(ckpt_logs_dir_mp)

    tb_logs_dir_mp = os.path.join(tb_logs_dir, modelparam)
    if not os.path.isdir(tb_logs_dir_mp):
        os.makedirs(tb_logs_dir_mp)

    if not get_dir_names_only:
        with open(os.path.join(
                        ckpt_logs_dir_mp, 'config.pkl'), 'wb') as f:
            pkl.dump(args, f)

        with open(os.path.join(text_logs_dir_mp,
                               modelparam+"_args.txt"), "a") as f:
            for arg in sorted(vars(args)):
                arg_str = str(arg) + ":  " + str(getattr(args, arg)) + "\n"
                f.write(arg_str)

    return text_logs_dir_mp, ckpt_logs_dir_mp, tb_logs_dir_mp


def make_modelparam_string_class(args):
    return ("class_%s,ar_%s,lrc_%.0E,wc_%d,dr_%.0E,CFLann_%d,"+\
           "num_ep_c%d,reg_%.0E,bn_%s,d_%s_%s_%s") % (
                                                args.classifier,
                                                args.architecture,
                                                args.learning_rate_class,
                                                args.weight_countfact_loss,
                                                args.decay_rate,
                                                args.cfl_annealing,
                                                args.num_epochs_class,
                                                args.lambda_reg,
                                                args.bn,
                                                args.dataset,
                                                args.time,
                                                args.fid)
