import vertexai
from mock import patch
from src.translator import translate_content

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_unexpected_language(mocker):
  # we mock the model's response to return a random message
  mocker.return_value.text = "I don't understand your request"
  content = "Aquí está su primer ejemplo." 
  # TODO assert the expected behavior
  assert translate_content(content) == (True, content)

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_empty_response(mocker):
  mocker.return_value = {}
  content = "Aquí está su primer ejemplo." 
  assert translate_content(content) == (True, content)

@patch('vertexai.preview.language_models._PreviewChatModel.start_chat')
def test_broken_chat_session(mocker):
  mocker.return_value = {}
  content = "Aquí está su primer ejemplo." 
  assert translate_content(content) == (True, content)

@patch('vertexai.preview.language_models._PreviewChatSession.send_message')
def test_malformed_response(mocker):
  mocker.return_value.text = "Here is your first example."
  del mocker.return_value.text
  content = "Aquí está su primer ejemplo." 
  assert translate_content(content) == (True, content)

@patch('vertexai.preview.language_models._PreviewChatModel.from_pretrained')
def test_broken_model(mocker):
  mocker.return_value = {}
  content = "Aquí está su primer ejemplo." 
  assert translate_content(content) == (True, content)
