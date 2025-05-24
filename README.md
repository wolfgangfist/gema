# CSM - Optimized Streaming/Finetuning Edition

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

Our fork adds **api**, **websocker** and **docker** to the [csm-streaming](https://github.com/davidbrowne17/csm-streaming) fork.

## 🚀 Additional Features in This Fork

- ✅ **REST API with FastAPI**  
  Easily generate speech via `/speak` endpoint using JSON requests.
- 🔐 **API Key Authentication**  
  Secure access to the API using `Authorization: Bearer <API_KEY>` header.
- 🐳 **Docker & Docker Compose Support**  
  Fully containerized setup with GPU access, Hugging Face token, and environment configuration.
- 🗂️ **OpenAPI schema**  
  Visit `/docs` for Swagger UI or `/openapi.json` for raw schema.

## 📘 Usage Documentation

The project includes a complete Docker setup located in the `Docker` folder with:

- `Dockerfile`
- `docker-compose.yaml`
- `.env`

When you run it using Docker Compose, it will:
- Automatically start the FastAPI server with Uvicorn
- Expose the API on `http://127.0.0.1:5537`
- Serve the interactive Swagger UI at `http://127.0.0.1:5537/docs`

You can access the OpenAPI schema and test endpoints from your browser.

### 🛠 Docker Setup Notes

Before starting, make sure to configure your `.env` file with these environment variables:

```env
BUILD_VERSION=0013
HUGGINGFACE_TOKEN=hf_000000
NO_TORCH_COMPILE=1
CSM_API_KEY=csmapikey_000000
```
✅ Important:
- Replace hf_000000 with your actual Hugging Face token to access the required models.
- Replace csmapikey_000000 with a secure random key. This will be used for authenticating API requests.


## 🛣️ Roadmap

- [x] Add REST API with FastAPI
- [x] Add API Key authentication with OpenAPI support
- [x] Add OpenAPI schema and Swagger docs
- [x] Add Docker & Docker Compose support
- [ ] Add support for **context-based generation** via the API
- [ ] Add **Postman collection** export
- [ ] Add runtime **volume and speaker variation controls**
- [ ] Add **streamed response** support

## FAQ

**Need help or additional doc?**

click here [csm-streaming](https://github.com/davidbrowne17/csm-streaming)

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Original Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.

## Streaming, Realtime Demo and Finetuning Implementation
David Browne

## Api, Websocket, Docker
Amir Nobandegan