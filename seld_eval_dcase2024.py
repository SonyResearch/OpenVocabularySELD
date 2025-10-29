# Copyright 2025 Sony AI

import codecs

from dcase2024_task3_seld_metrics import parameters, cls_compute_seld_results


def all_seld_eval(ref_files_folder, pred_output_format_files, test_dataset, part_category_txt=None,
                  print_stdout=True, result_path=None):
    params = parameters.get_params()
    if test_dataset == "STARSS23":
        params['unique_classes'] = 13
        list_part_category = None
    elif test_dataset == "TNSSE21":
        params['unique_classes'] = 12
        list_part_category = None
    elif test_dataset == "FSD50K_TAU-SRIR_part":
        params['unique_classes'] = 12
        if part_category_txt is not None:
            with open(part_category_txt, 'r') as file:
                list_part_category = [int(line.strip()) for line in file.readlines()]
        else:
            assert False, "FSD50K_TAU-SRIR_part requires to set part_category_txt"

    score_obj = cls_compute_seld_results.ComputeSELDResults(params, ref_files_folder=ref_files_folder, list_part_category=list_part_category)
    er20, f20, le, lr, seld_err, classwise_test_scr, other_scores = score_obj.get_SELD_Results(pred_output_format_files)
    er20_d, er20_i, er20_s, pre, rec, lf, lp, classwise_other_results = other_scores

    if print_stdout is True:
        print('SELD scores')
        print('All\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP')
        print('All\t{:0.3f}\t{:0.3f}\t{:0.2f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(
            er20, f20, le, lr, seld_err, er20_d, er20_i, er20_s, pre, rec, lf, lp))
        if params['average'] == 'macro':
            print('Class-wise results')
            print('Class\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP')
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                    cls_cnt,
                    classwise_test_scr[0][cls_cnt],
                    classwise_test_scr[1][cls_cnt],
                    classwise_test_scr[2][cls_cnt],
                    classwise_test_scr[3][cls_cnt],
                    classwise_test_scr[4][cls_cnt],
                    classwise_other_results[0][cls_cnt],
                    classwise_other_results[1][cls_cnt],
                    classwise_other_results[2][cls_cnt],
                    classwise_other_results[3][cls_cnt],
                    classwise_other_results[4][cls_cnt],
                    classwise_other_results[5][cls_cnt],
                    classwise_other_results[6][cls_cnt]))

    if result_path is not None:
        print('SELD scores',
              file=codecs.open(result_path, 'w', 'utf-8'))
        print('All\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP',
              file=codecs.open(result_path, 'a', 'utf-8'))
        print('All\t{:0.3f}\t{:0.3f}\t{:0.2f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(
            er20, f20, le, lr, seld_err, er20_d, er20_i, er20_s, pre, rec, lf, lp),
            file=codecs.open(result_path, 'a', 'utf-8'))
        if params['average'] == 'macro':
            print('Class-wise results',
                  file=codecs.open(result_path, 'a', 'utf-8'))
            print('Class\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP',
                  file=codecs.open(result_path, 'a', 'utf-8'))
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                    cls_cnt,
                    classwise_test_scr[0][cls_cnt],
                    classwise_test_scr[1][cls_cnt],
                    classwise_test_scr[2][cls_cnt],
                    classwise_test_scr[3][cls_cnt],
                    classwise_test_scr[4][cls_cnt],
                    classwise_other_results[0][cls_cnt],
                    classwise_other_results[1][cls_cnt],
                    classwise_other_results[2][cls_cnt],
                    classwise_other_results[3][cls_cnt],
                    classwise_other_results[4][cls_cnt],
                    classwise_other_results[5][cls_cnt],
                    classwise_other_results[6][cls_cnt]),
                    file=codecs.open(result_path, 'a', 'utf-8'))

    return er20, f20, le, lr, seld_err, other_scores
