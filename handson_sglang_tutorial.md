# Hands-on Tutorial for SGLang

This tutorial will guide you through the steps to set up and run SGLang on a cloud instance with GPU support. We will use an NVIDIA-sponsored cloud service to get access to a powerful GPU instance, set up the SGLang environment using Docker, and launch a SGLang server with a pre-trained model.

Highly recommend to follow first 2 steps to get your instance and set up the sglang docker image before the tutorial session starts, so that you could have more time to explore sglang during the session.

## First Step: getting your instance

Thanks to NVIDIA's sponsorship, you can get access to a free cloud instance here:

1. Follow this link to join get the access to NVIDIA Brev for this tutorial:

    [link to join NVIDIA Brev](https://brev.nvidia.com/invite?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHBpcmF0aW9uIjoxNzYxODYxODEyLCJvcmdJZCI6Im9yZy0zNDROYWZyQlVaUUpTQXJOclVlcUdQTWdDVVAiLCJ1c2VySWQiOiJ1c2VyLTM0MXY5ajZVa2VYUFpqemRoSXNPbDVVUXNNRSJ9.hJ5jA8apd1k6HbNcUmX-sfMKPybbc1_JAFgXLjGmKac&orgID=org-344NafrBUZQJSArNrUeqGPMgCUP)

2. Create an account if you don't have one already.
    ![login](login.png)

    Setup the account name:
    ![account_name](account_name.png)

    Then you will be redirected to team's dashboard where you can create a new instance:
    ![create new instance](create_new_instance.png)

3. Click on "Create New Instance", then select from the list of machines.

    ![list of h100s](h100s.png)

    If H100 is not available, you can select an A100 instance instead.

4. Name your instance with your name and click on "Deploy".
    ![deploy](deploy.png)
    Wait for the instance and the "VM Mode" to be ready (it may take a few minutes).
    Then you could see the "Open Notebook" button available.
    ![instance ready](instance_ready.png)

Congratulations! You now have your cloud instance ready.

## Second Step: Setting up sglang docker image

1. Click on "Open Notebook", it will open a new tab in your browser.
    You will see a JupyterLab interface, then open a terminal in JupyterLab.
    <!-- ![open terminal](open_terminal.png) -->

2. In the terminal, run the following command to verify the GPU is available:
    ```bash
    nvidia-smi
    ```
    You should see the GPU information displayed.
    ![nvidia-smi](nvidia-smi.png)

3. Then pull the sglang docker image (This may take a few minutes):
    ```bash
    docker pull lmsysorg/sglang:latest
    ```

## Third Step: Launching sglang server with gpt-oss-20b model
1. In the terminal, run the following command to start a sglang container (replace the hf token with yours):
    ```bash
    docker run -itd --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=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" \
    --ipc=host \
    --name sglang_gpt_oss_server \
    lmsysorg/sglang:latest \
    /bin/zsh
    ```

2. Then exec into the running container:
    ```bash
    docker exec -it sglang_gpt_oss_server /bin/zsh
    ```

3. (Optional) Download the gpt-oss-20b model in advance to speed up the server launch, or the model will be downloaded automatically when launching the server for the first time.
    ```bash
    hf download openai/gpt-oss-20b
    ```

4. Run the following command to launch a gpt-oss-20b sglang server:
    ```bash
    python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --reasoning-parser gpt-oss --tool-call-parser gpt-oss --host 0.0.0.0
    ```

5. You should see the server is running and listening on port 30000:
    ![server ready](server_ready.png)

6. Send a test request to the server using curl from another terminal:
    ```bash
    curl -X POST http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 50
    }'
    ```
    You should receive a response with the answer:
    ![model answer](model_answer.png)

Congratulations! You have successfully set up and launched the SGLang server with the gpt-oss-20b model on your cloud instance.

## Fourth Step: Agentic usage of SGLang

To demonstrate the agentic capabilities of SGLang, we will use [qwen-code](https://github.com/QwenLM/qwen-code) as the platform to interact with the SGLang server we just set up.

1. (Optional) Run the following commands to exec into the sglang container: (the purpose of installing the qwen-code in the sglang container is you can use qwen-code's inbuilt tools to explore the sglang repo, like asking for code explanation etc.)
    ```bash
    docker exec -it sglang_gpt_oss_server /bin/zsh
    ```

2. You need to install the node.js first following this [link](https://nodejs.org/en/download).
    Or just run the following commands in the terminal:
    ```bash
    # Download and install nvm:
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

    # in lieu of restarting the shell
    \. "$HOME/.nvm/nvm.sh"

    # Download and install Node.js:
    nvm install 22

    # Verify the Node.js version:
    node -v # Should print "v22.21.0".

    # Verify npm version:
    npm -v # Should print "10.9.4".
    ```

3. Install the qwen-code:
    ```bash
    npm install -g @qwen-code/qwen-code@latest
    qwen --version
    ```
    You should see the qwen-code version displayed.

4. Important: Set the endpoint environment variable to point to your SGLang server:
    ```bash
    export OPENAI_API_KEY="sk-123456" # dummy key
    export OPENAI_BASE_URL="http://localhost:30000/v1"
    export OPENAI_MODEL="openai/gpt-oss-20b"
    ```

5. Now you can use qwen-code to interact with the SGLang server. (You may better change the theme of jupyterlab to "dark" for better visibility by Settings >> Theme >> JupyterLab Dark.)
    ```bash
    qwen
    ```
    For example, type `/init` to let the qwen-code analyze the current directory. You can see several tools are called.
    ![qwen-code-init](qwen_code_init.png)

## Advanced Topic: Using speculative decoding with SGLang

