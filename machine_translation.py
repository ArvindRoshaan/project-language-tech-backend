import subprocess

language_code = {
    "English": "eng",
    "Lambani": "lmn",
    "Kannada": "kan",
    "Tamil": "ta"
}

language_dict = {
    "eng": "English",
    "en": "English",
    "lmn": "Lambani",
    "kan": "Kannada",
    "ta": "Tamil"
}

available_targets = {"eng": ["lmn"], "kan":["lmn"], "lmn":["eng", "kan"], "en":["ta"]}

available_models = {"eng-lmn", "lmn-eng", "kan-lmn", "lmn-kan", "en-ta"}

def translate(text, src_full, tgt_full):
    generated_text = ""

    input_file_path = "../../nmt/fairseq/inference_scripts"
    input_file_name = "inference.txt"

    print(text)
    print(src_full)
    print(tgt_full)

    src_full = src_full.strip()
    tgt_full = tgt_full.strip()
    src = language_code[src_full.strip()]
    tgt = language_code[tgt_full.strip()]

    if tgt == "ta":
        src = "en"

    if f"{src}-{tgt}" not in available_models:
        return f"{src_full} to {tgt_full} translation system is not available"

    with open(f"{input_file_path}/{input_file_name}", 'w') as f:
        f.write(text.strip()+"\n")

    # Path to your shell script
    if tgt == 'lmn' and src == 'eng':
        script_path = "../../nmt/fairseq/inference_scripts/machine_translation_freezed_inference.py"
    elif tgt == 'ta' and src == 'en':
        script_path = "../../nmt/fairseq/inference_scripts/machine_translation_inference_en_ta.py"
    else:
        script_path = "../../nmt/fairseq/inference_scripts/machine_translation_baseline_inference.py"
    
    print(f"the script run is {script_path}")
    result = subprocess.run(['python', script_path, src, tgt, input_file_path, input_file_name])
    # Call the shell script using subprocess
    #result = subprocess.check_call([script_path, '-s', src, '-t', tgt, '-p', input_file_path, '-n', input_file_name])
    #, capture_output=True, text=True
    #bash machine_translation_baseline_inference.sh -s eng -t lmn -p /raid/ai21mtech12004/workspace/nmt/fairseq/inference_scripts -n inference.txt
    
    # Check the result
    if result.returncode == 0:
        print("Script executed successfully")
        #print("Output:", result.stdout)
    else:
        print("Script failed with return code:", result.returncode)
        print("Error:", result.stderr)

    output_file_name = 'machine_translation_output.txt'
    
    with open(f"{input_file_path}/{output_file_name}", 'r') as f:
        out = f.readlines()
    out = out[0].strip()

    return out

#print(translate("I am in Bangalore!", "English", "Tamil"))
#print(translate("I would linger near the temple.", "English", "Lambani"))