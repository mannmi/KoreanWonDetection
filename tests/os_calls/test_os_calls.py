import unittest
# Add the src directory to the Python path
from unittest.mock import patch, mock_open
from src.os_calls.oscalls import is_running_in_docker, check_cuda_available


# Assuming the functions and class are in a module named `module_name`


class TestFunctions(unittest.TestCase):

    @patch('os.path.exists')
    def test_is_running_in_docker_dockerenv(self, mock_exists):
        # Test when /.dockerenv exists
        mock_exists.return_value = True
        self.assertTrue(is_running_in_docker())

        # Test when /.dockerenv does not exist
        mock_exists.return_value = False
        with patch('builtins.open', mock_open(read_data='')) as mock_file:
            self.assertFalse(is_running_in_docker())
            mock_file.assert_called_once_with('/proc/1/cgroup', 'rt')

    @patch('builtins.open', new_callable=mock_open, read_data='docker')
    def test_is_running_in_docker_cgroup(self, mock_file):
        # Test when 'docker' is in /proc/1/cgroup
        self.assertTrue(is_running_in_docker())
        mock_file.assert_called_once_with('/proc/1/cgroup', 'rt')

    @patch('builtins.open', new_callable=mock_open, read_data='')
    def test_is_running_in_docker_cgroup_not_docker(self, mock_file):
        # Test when 'docker' is not in /proc/1/cgroup
        self.assertFalse(is_running_in_docker())
        mock_file.assert_called_once_with('/proc/1/cgroup', 'rt')

    @patch('torch.cuda.is_available', return_value=True)
    def test_check_cuda_available_true(self, mock_cuda_available):
        self.assertTrue(check_cuda_available())
        mock_cuda_available.assert_called_once()

    @patch('torch.cuda.is_available', return_value=False)
    def test_check_cuda_available_false(self, mock_cuda_available):
        self.assertFalse(check_cuda_available())
        mock_cuda_available.assert_called_once()


if __name__ == '__main__':
    unittest.main()
