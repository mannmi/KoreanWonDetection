import os
import sys
import unittest
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config_loader.configLoader import YmlLoader


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        if os.getenv('GITHUB_ACTIONS'):
            conf = "./tests/yaml_loader/TestData/config.yml"
            test_file = "./tests/yaml_loader/TestData/tmpConfig.yml"
        else:
            conf = "./TestData/config.yml"
            test_file = "./TestData/tmpConfig.yml"

        if not os.path.exists(conf):
            raise FileNotFoundError(f"Config file not found: {conf}")

        os.system(f'cp {conf} {test_file}')
        self.ymlLoader = YmlLoader(test_file)

    def test_yml_loader_data(self):
        dataRead = self.ymlLoader.data
        refString = {
            'System': {
                'ProjectRoot': 'F:\\PycharmProjects\\TPRO',
                'chunksize': '10 ** 1',
                'delimiter': ';',
                'encoding': 'utf-8',
                'lineterminator': '\\n',
                'projectRoot_file': 'appDemoAsync.py'
            },
            'acceskeys': {'HUGGINGFACE_TOKEN': 'hf_testtoken'},
            'bertModel': [
                {
                    'filter': {
                        'model_name': 'unitary/unbiased-toxic-roberta',
                        'model_path': '/models/filtration_model'
                    }
                }
            ],
            'combinedModel': {
                'contains': {
                    '-gpt2': {
                        'model_name': 'joined-Model',
                        'model_path': 'models/joined-Model'
                    }
                }
            },
            'gpt2Model': [
                {
                    'gpt2': {
                        'GPT2LMHeadModel': 'gpt2',
                        'GPT2Tokenizer': 'gpt2',
                        'model_name': 'gpt2',
                        'model_path': 'models/gpt2_model'
                    }
                },
                {
                    'gpt2-german': {
                        'GPT2LMHeadModel': 'dbmdz/german-gpt2',
                        'GPT2Tokenizer': 'dbmdz/german-gpt2',
                        'model_name': 'dbmdz/german-gpt2',
                        'model_path': 'models/german-gpt2'
                    }
                }
            ]
        }
        self.assertEqual(dataRead, refString)

    def test_yml_loader_keyAcces(self):
        dataRead = self.ymlLoader.data['System']['delimiter']
        refString = ';'
        self.assertEqual(dataRead, refString)

    def test_yml_loader_changeKey(self):
        dataRead = self.ymlLoader.data['System']['delimiter']
        refString = ';'
        self.assertEqual(dataRead, refString)


if __name__ == '__main__':
    unittest.main()
