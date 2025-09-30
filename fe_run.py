
from fe_utils import generar_features_new, generar_sef_new, generar_features_and_sef
import time



if __name__ == "__main__":
    aux_path = ['edf/367-01_intraoperatorio_arm_1_archivo_edf.zip',
    'edf/367-02_intraoperatorio_arm_1_archivo_edf.zip',
    'edf/367-03_intraoperatorio_arm_1_archivo_edf.zip',
    'edf/367-04_intraoperatorio_arm_1_archivo_edf.zip',
    'edf/367-05_intraoperatorio_arm_1_archivo_edf.zip']

    save_path = 'csvfiles/'

    start = time.time()
    aux_df2 = generar_features_and_sef(aux_path, overlap = 0.9, window_size_seg=10, peuma1=False, edf=True, aux_save_path=save_path, report_performance=False)
    print("Tiempo total de ejecución (features):", time.time() - start)
    # aux_df2_sef = generar_sef_new(aux_path, overlap = 0.9, window_size_seg=10, peuma1=False, edf=True, aux_save_path=save_path)
    # print("Tiempo total de ejecución (SEF):", time.time() - start)