import argparse
import copy
import math  # This is needed!
import os
import pickle


def reader(filename):
    for row in open(filename, "r"):
        yield row.strip()


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def parse_outs(exp_folder, steps, outf):
    step_list = eval(steps)
    print(0, step_list)
    for step in step_list:

        pred_file = os.path.join(exp_folder, 'predictions', f'off_policy_valid-{step}.txt')
        evict_file = os.path.join(exp_folder, 'evictions', f'valid-{step}.txt')
        output_folder = os.path.join(outf, exp_folder.split('/')[-1])
        os.makedirs(os.path.join(output_folder, str(step)), exist_ok=True)
        output_file = os.path.join(output_folder, str(step), 'valid-parsed-preds.txt')

        print(111, step, pred_file, evict_file, output_folder, output_file)
        pred_reader = reader(pred_file)
        evict_reader = reader(evict_file)

        i = 0

        # pred_lines = 0
        # evict_lines = 0
        # for pred_line in pred_reader:
        #     if pred_line == "":
        #         pred_lines += 1
        # for evict_line in evict_reader:
        #     evict_lines += 1
        #
        # print(10, pred_lines, evict_lines)

        in_cache_line = False
        set_dict = {}
        instance_dict = {}
        while True:
            pred_line = next(pred_reader, None)
            if pred_line is None:
                print('Iterated through the pred file.')
                break
            if 'PC' in pred_line:
                instance_dict['pc'] = pred_line.split(' ')[1]
            if 'Address' in pred_line:
                instance_dict['address'] = pred_line.split(' ')[1]
            if 'Cache lines' in pred_line:
                in_cache_line = True
                instance_dict['cache_lines_pc'] = []
                instance_dict['cache_lines_address'] = []
                instance_dict['cache_lines_pred_rank'] = []
                instance_dict['cache_lines_prob'] = []
                instance_dict['cache_lines_reuse_distance'] = []
            if 'Attention' in pred_line:
                in_cache_line = False
            if in_cache_line and pred_line[:3] == '|  ':
                cache_line = pred_line.replace(' ', '').split('|')
                instance_dict['cache_lines_pc'].append(cache_line[2])
                instance_dict['cache_lines_address'].append(cache_line[3])
                instance_dict['cache_lines_pred_rank'].append(eval(cache_line[4]))
                instance_dict['cache_lines_prob'].append(eval(cache_line[5]))
                instance_dict['cache_lines_reuse_distance'].append(eval(cache_line[7]))

            if pred_line == "":
                evict_line = eval(next(evict_reader).replace('Infinity', 'math.inf').replace('false', 'False').replace('true', 'True'))
                instance_dict['evict'] = evict_line['evict']

                assert instance_dict['pc'] == evict_line['pc'], f"PC does not match between pred ({instance_dict['pc']}) and evict ({evict_line['pc']}) file."
                assert instance_dict['address'] == evict_line[
                    'address'], f"Address does not match between pred ({instance_dict['address']}) and evict ({evict_line['address']}) file."
                if evict_line['set_id'] in set_dict:
                    set_dict[evict_line['set_id']].append(copy.deepcopy(instance_dict))
                else:
                    set_dict[evict_line['set_id']] = [copy.deepcopy(instance_dict)]
                i += 1
                if i % 10000 == 0:
                    print(i)
                # print(f"instance_dict: {instance_dict} \n")
            # print(f"set_dict: {set_dict} \n")
        print("Saving..")
        save_obj(set_dict, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse output files")
    parser.add_argument('--exp-folder', required=True)
    parser.add_argument('--steps', type=str, required=True, help="For example '[1000,2000]'")
    parser.add_argument('--out-folder', required=True)

    args = parser.parse_args()

    parse_outs(args.exp_folder, args.steps, args.out_folder)
