import argparse
import copy
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model
import report_man
from dataloader import SpectrumLoader


class Metric:
    def __init__(self):
        self.truepositve = 0
        self.truenegative = 0
        self.falsepositve = 0
        self.falsenegative = 0

    def __str__(self):
        F1score, total_tests, accuracy, precision, recall = self.get_score()

        infostring = 'Acc: %.3f%% (%d: TP %d TN %d FP %d FN %d), P: %.3f%%, R: %.3f%%, F1:%.3f%%' % (
            100.*accuracy, total_tests,
            self.truepositve, self.truenegative, self.falsepositve, self.falsenegative,
            100.*precision, 100.*recall, 100.*F1score)
        return infostring

    def get_score(self, F1_only=False):
        total_correct = self.truepositve + self.truenegative
        total_tests = self.truepositve+self.truenegative + \
            self.falsepositve+self.falsenegative
        accuracy = float(total_correct) / (float(total_tests)+1e-6)
        precision = float(self.truepositve) / \
            (float(self.truepositve)+float(self.falsepositve)+1e-6)
        recall = float(self.truepositve) / \
            (float(self.truepositve)+float(self.falsenegative)+1e-6)
        F1score = 2. * (precision * recall) / (precision + recall+1e-6)
        if F1_only:
            return F1score
        else:
            return F1score,  total_tests, accuracy, precision, recall

    def update(self, regressor, target_regress):
        self.truepositve += torch.sum(
            (abs(regressor-target_regress) < 0.5/2) & (abs(target_regress - 0.5) <= 0.5/2))
        self.truenegative += torch.sum(
            (abs(regressor-target_regress) < 0.5/2) & (abs(target_regress - 0.5) > 0.5/2))
        self.falsepositve += torch.sum(
            (abs(regressor-target_regress) >= 0.5/2) & (abs(target_regress - 0.5) > 0.5/2))
        self.falsenegative += torch.sum(
            (abs(regressor-target_regress) >= 0.5/2) & (abs(target_regress - 0.5) <= 0.5/2))


def main():
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(
        description='Locate events from SMFS datastream')

    parser.add_argument('--train',  default=1, type=int,
                        help='train flag')
    parser.add_argument('--minibatches_per_step',
                        default=10, type=int, help='minibatches_per_step')
    parser.add_argument('--minibatch_size',
                        default=300, type=int, help='minibatch_size')
    parser.add_argument('--epoch',
                        default=31, type=int, help='epoch')
    parser.add_argument('--learning_rate',
                        default=0.001, type=float, help='learning_rate')
    parser.add_argument('--data_split',
                        default="0,1.0", type=str, help='data_split')
    parser.add_argument('--data_kept',
                        default="0,0", type=str, help='data_split')
    parser.add_argument('--noiselevel',
                        default="0,0", type=str, help='noiselevel')
    parser.add_argument('--datafolder',
                        default="./output", type=str, help='datafolder for .np files')
    parser.add_argument('--modelfile',
                        default="./report/model-latest.pt", type=str, help='modelfile for a .pt file')
    parser.add_argument('--cuda', metavar='1 or 0', default=0 if torch.cuda.is_available() else 0, type=int,
                        help='cuda')
    parser.add_argument('--downsampling',  default=1, type=int,
                        help='downsampling')
    parser.add_argument('--pdfheader',  default="test", type=str,
                        help='pdfheader')
    parser.add_argument('--test_size',
                        default=300, type=int, help='test_size')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda else 'cpu')
    print(device)

    data_split = [float(item) for item in args.data_split.split(',')]
    data_kept = [int(item) for item in args.data_kept.split(',')]
    if args.cuda == 0:
        print("WARNING: run in debugging mode")
        args.epoch = 0

    np.random.seed(0)
    torch.manual_seed(0)
    if not args.train:
        print("WARNING: Running in eval mode, using no noise!")
        noiselevel = [0, 0]
    else:
        args.pdfheader = "train"
        noiselevel = [float(item) for item in args.noiselevel.split(',')]
        print("using noise level "+str(noiselevel))

    spLoader = SpectrumLoader(
        datafolder=args.datafolder, recpetive_field=model.receptive_field, filename_suffix="", AWGN=noiselevel, downsampling=args.downsampling, data_split=data_split, data_kept=data_kept)
    print("DataSet Size: %d" % spLoader.size)
    total_steps_per_epoch = (
        spLoader.size+args.minibatches_per_step) // args.minibatches_per_step
    print("total_steps_per_epoch: %d" % total_steps_per_epoch)

    seq_predictor = model.Sequence(2).to(device)
    if not args.train:
        seq_predictor.load_state_dict(torch.load(
            args.modelfile, map_location=device))
        seq_predictor.eval()
        print("loading trained model")
    if args.cuda == 0:
        from torchsummary import summary
        summary(seq_predictor, (2, 224))

    seq_predictor.double()
    bcecriterion = nn.BCELoss()
    msecriterion = nn.L1Loss()
    if args.train:

        optimizer = optim.Adam(seq_predictor.parameters(),
                               lr=args.learning_rate)  # , momentum=0.9
    best_metric_epoch = Metric()
    best_epoch_id = 0

    for epoch in range(args.epoch):
        metric_epoch = Metric()
        spLoader.reset(randomize=True)

        for step in range(total_steps_per_epoch):
            if args.train:
                optimizer.zero_grad()
            total_loss = 0
            total_mse_loss = 0
            for _ in range(args.minibatches_per_step):
                source_minibatch, target_regress, target_invreg, _ = spLoader.get_samples(
                    device=device, sample_per_file=args.minibatch_size)

                predict_invreg, predict_regressor = seq_predictor(
                    source_minibatch)

                # lossbce = bcecriterion(predict_binary, target_binary)
                lossmse = msecriterion(
                    predict_regressor, target_regress)*0.5
                metric_epoch.update(predict_regressor, target_regress)
                lossmse_inv = msecriterion(
                    predict_invreg, target_invreg)*0.5
                # metric_epoch.update(predict_invreg, target_invreg)
                loss = lossmse+lossmse_inv
                total_loss += float(loss.item())
                total_mse_loss += float(lossmse.item())
                if args.train:
                    loss /= args.minibatches_per_step
                    loss.backward(retain_graph=True)
            if args.train:
                optimizer.step()

                report_man.progress_bar(step, total_steps_per_epoch, args.pdfheader +
                                        " Epoch: %d | Loss: %.6f/%.6f | " % (epoch, total_loss, total_mse_loss) + str(metric_epoch))

        print("FINAL: "+args.pdfheader+" | Epoch: %d | total metric " %
              (epoch) + str(metric_epoch))
        if metric_epoch.get_score(F1_only=True) >= best_metric_epoch.get_score(F1_only=True):
            best_metric_epoch = metric_epoch
            best_epoch_id = epoch

        with torch.no_grad():
            if not args.train:
                print("No checkpointing to prevent trouble.")
            else:
                if epoch % 10 == 0:
                    print("Checkpoint ... | Epoch: %d" % (epoch))
                    torch.save(seq_predictor.state_dict(
                    ), report_man.get_report_folder()+'model%d.pt' % epoch)

    with open(report_man.get_report_folder()+args.pdfheader+".log", "a") as logfile:
        import sys
        logfile.write(' '.join(sys.argv)+"\n")
        logfile.write('at {}: {}\n'.format(
            best_epoch_id, str(best_metric_epoch)))

    if args.train:
        torch.save(seq_predictor.state_dict(), args.modelfile)
    else:
        test_on_single(args, spLoader, device,
                       seq_predictor, noiselevel, args.pdfheader)

    with open(report_man.get_report_folder()+args.pdfheader+".log", "r") as logfile:
        print(logfile.read())


def test_on_single(args, spLoader, device, seq_predictor, noiselevel, pdfheader):
    try:
        with open(report_man.get_report_folder()+args.pdfheader+".json", "r") as jsonfile:
            summerize_results = json.load(jsonfile)
    except Exception as e:
        print("ERROR: fail to load result json "+str(e))
        summerize_results = {}
    for print_id in range(min(args.test_size, spLoader.size)):
        source, _, _, rawinfo = spLoader.get_samples(
            device=device)
        (source_raw, target_raw, target_events, filename) = rawinfo
        predict_invreg_i_list = []
        predict_regress_i_list = []

        for i_start in range(0, source.size(0), args.minibatch_size):
            i_end = min(i_start+args.minibatch_size, source.size(0))
            source_minibatch = source[i_start: i_end]
            predict_invreg_i, predict_regression_i = seq_predictor(
                source_minibatch)
            predict_invreg_i_list.append(predict_invreg_i.detach())
            predict_regress_i_list.append(predict_regression_i.detach())

        predict_regressor = torch.cat(predict_regress_i_list, 0)
        predict_invreg = torch.cat(predict_invreg_i_list, 0)

        predict_regressor = predict_regressor.cpu().numpy()
        predict_invreg = predict_invreg.cpu().numpy()

        result_filename = report_man.get_report_folder() + pdfheader+"-" + \
            os.path.basename(filename)
        np.save(result_filename, np.vstack(
            (predict_regressor, predict_invreg)))

        use_differential = True
        predict_events = []

        if not use_differential:

            predict_regressor = (abs(predict_regressor-0.5) *
                                 (predict_regressor > 0.01))*2.0
            supression_size = 10
            for reg_i in range(supression_size, predict_regressor.shape[0]-supression_size):
                for supression_range_i in range(reg_i-supression_size, reg_i+supression_size):

                    if predict_regressor[supression_range_i] < predict_regressor[reg_i]:
                        break
                    if predict_regressor[supression_range_i] > 0.5:
                        break
                    if predict_regressor[supression_range_i] < 1e-6:
                        break
                else:
                    predict_events.append(
                        (reg_i+model.receptive_field//2, math.exp(-1.0*abs(predict_regressor[reg_i]))))
        else:
            predict_regressor = predict_regressor * \
                (predict_regressor > 0.01) - \
                predict_invreg * (predict_regressor > 0.01)

            supression_size = 5
            for reg_i in range(supression_size, predict_regressor.shape[0]-supression_size):
                left_avg = np.mean(
                    predict_regressor[reg_i-supression_size:reg_i])
                right_avg = np.mean(
                    predict_regressor[reg_i+1:reg_i+supression_size])
                if left_avg < 0 and right_avg > 0:
                    predict_events.append(
                        (reg_i+model.receptive_field//2, math.exp(-1.0*abs(predict_regressor[reg_i]))))

        summerize_results[filename] = (target_events, predict_events)
        if predict_events:
            predict_events = np.array(predict_events).transpose()[0].tolist()

            detection_to_target_distances = spLoader.get_distances(
                target_events, predict_events)
            target_to_detection_distances = spLoader.get_distances(
                predict_events, target_events)

            print(pdfheader+" {} | Print: {} d2t: {}  t2d: {}" .format(print_id, filename,
                                                                       np.mean(detection_to_target_distances), np.mean(target_to_detection_distances)))
            if print_id < 30:
                draw_pdf(result_filename, ' AWGN '+str(noiselevel),
                         predict_regressor, predict_events, target_raw, source_raw)
            else:
                print("Ignore PDF printing.")
        else:
            print(
                pdfheader+" {} | Print: {}, nothing detected." .format(print_id, filename))
    with open(report_man.get_report_folder()+args.pdfheader+".json", "w") as jsonfile:
        json.dump(summerize_results, jsonfile, indent=4)


def draw_pdf(pdfheader, noiselevel, predict_regressor, predict_events, target_raw, source_raw):
    def draw(data, color, offset=0, linewidth=2.0):
        plt.plot(np.arange(data.shape[0]) +
                 offset, data, color, linewidth=linewidth)
    # plotting and display
    plt.figure(figsize=(30, 10))
    plt.xlabel(pdfheader+noiselevel, fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    for i in predict_events:
        plt.axvline(i, color='purple', linestyle='--')
    draw(source_raw[0], 'y')
    draw(source_raw[1], 'b')
    draw(target_raw, 'g')
    draw(predict_regressor, 'black', offset=model.receptive_field//2, linewidth=1.0)
    plt.savefig(pdfheader + ".pdf")
    plt.close()


if __name__ == '__main__':
    main()
