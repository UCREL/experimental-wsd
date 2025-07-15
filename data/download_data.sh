# The data has come from http://lcl.uniroma1.it/wsdeval/
wsd_english_eval_2017="WSD_Evaluation_Framework"
wget -O "${wsd_english_eval_2017}.zip" http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip "${wsd_english_eval_2017}.zip"
rm "${wsd_english_eval_2017}.zip"

# The data has come from this website https://sapienzanlp.github.io/xl-wsd/docs/data/
# which is where you can find the original Google drive link
xl_wsd_2021="xl-wsd"
wget -O "${xl_wsd_2021}.zip" "https://drive.usercontent.google.com/download?id=19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0&export=download&confirm=xxx"
unzip "${xl_wsd_2021}.zip"
rm "${xl_wsd_2021}.zip"

# The data has come from https://github.com/SapienzaNLP/wsd-hard-benchmark
english_hard_wsd="english-hard-wsd"
wget -O "${english_hard_wsd}.zip" https://github.com/SapienzaNLP/wsd-hard-benchmark/archive/refs/heads/main.zip
unzip "${english_hard_wsd}.zip"
rm "${english_hard_wsd}.zip"
mkdir "${english_hard_wsd}"
mv ./wsd-hard-benchmark-main/wsd_hard_benchmark/* $english_hard_wsd/.
rm -r ./wsd-hard-benchmark-main

# Gather the English WordNet
uv run python -m wn download omw-en:1.4

# Adds the data folder paths to the environment file.
script_directory=$(realpath $(dirname ${BASH_SOURCE[0]}))
env_file="${script_directory}/../.env"
grep -q "XL_WSD_PATH" $env_file

if [ $? -eq 0 ]
then
    echo "Not adding Environment variables to the ${env_file} as they already exist"
    exit 0
fi

echo "Adding environment variables to the ${env_file}"

xl_wsd_path="${script_directory}/xl-wsd"
english_hard_wsd_path="${script_directory}/english-hard-wsd"
english_raganato_path="${script_directory}/WSD_Evaluation_Framework"
echo -e "\nXL_WSD_PATH=${xl_wsd_path}\nENGLISH_MARU_HARD=${english_hard_wsd_path}\nENGLISH_RAGANATO=${english_raganato_path}" >> $env_file