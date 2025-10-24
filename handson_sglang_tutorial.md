# Hands-on Tutorial for SGLang

This tutorial will guide you through the steps to set up and run SGLang on a cloud instance with GPU support. You'll use an NVIDIA-sponsored cloud service to access a powerful GPU instance, set up the SGLang environment using Docker, and launch an SGLang server with a pre-trained model.

**Highly recommended**: Complete the first two steps (getting your instance and setting up the SGLang Docker image) before the tutorial session starts, so you have more time to explore SGLang during the session.

## Step 1: Get Your Cloud Instance

Thanks to NVIDIA's sponsorship, you can get access to a free cloud instance.

1. Follow this link to join NVIDIA Brev for this tutorial:

    [link to join NVIDIA Brev](https://brev.nvidia.com/invite?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHBpcmF0aW9uIjoxNzYxODYxODEyLCJvcmdJZCI6Im9yZy0zNDROYWZyQlVaUUpTQXJOclVlcUdQTWdDVVAiLCJ1c2VySWQiOiJ1c2VyLTM0MXY5ajZVa2VYUFpqemRoSXNPbDVVUXNNRSJ9.hJ5jA8apd1k6HbNcUmX-sfMKPybbc1_JAFgXLjGmKac&orgID=org-344NafrBUZQJSArNrUeqGPMgCUP)

2. Create an account if you don't have one already.
    ![login](login.png)

    Setup the account name:
    ![account_name](account_name.png)

    Then you will be redirected to team's dashboard where you can create a new instance:
    ![create new instance](create_new_instance.png)

3. Click "Create New Instance", then select from the list of available machines.

    ![list of h100s](h100s.png)

    **Note**: If H100 is not available, you can select an A100 instance instead.

4. Name your instance (use your name) and click "Deploy".
    
    ![deploy](deploy.png)
    
    Wait for the instance and "VM Mode" to be ready (this may take a few minutes). Once ready, you'll see the "Open Notebook" button become available.
    
    ![instance ready](instance_ready.png)

**Congratulations!** Your cloud instance is now ready.

## Step 2: Set Up the SGLang Docker Image

1. Click "Open Notebook". This will open a new tab in your browser with a JupyterLab interface. Open a terminal in JupyterLab.
    <!-- ![open terminal](open_terminal.png) -->

2. In the terminal, run the following command to verify that the GPU is available:
    ```bash
    nvidia-smi
    ```
    You should see the GPU information displayed.
    
    ![nvidia-smi](nvidia-smi.png)

3. Pull the SGLang Docker image (this may take a few minutes):
    ```bash
    docker pull lmsysorg/sglang:latest
    ```

## Step 3: Launch the SGLang Server with the gpt-oss-20b Model

1. In the terminal, run the following command to start an SGLang container (replace the HF token with yours):
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

2. Enter the running container:
    ```bash
    docker exec -it sglang_gpt_oss_server /bin/zsh
    ```

3. **(Optional)** Download the gpt-oss-20b model in advance to speed up the server launch. If you skip this step, the model will be downloaded automatically when you launch the server for the first time.
    ```bash
    hf download openai/gpt-oss-20b
    ```

4. Run the following command to launch the gpt-oss-20b SGLang server:
    ```bash
    python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --reasoning-parser gpt-oss --tool-call-parser gpt-oss --host 0.0.0.0
    ```

5. You should see the server running and listening on port 30000:
    
    ![server ready](server_ready.png)

6. Send a test request to the server using `curl` from another terminal:
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

**Congratulations!** You have successfully set up and launched the SGLang server with the gpt-oss-20b model on your cloud instance.

## Step 4: Agentic Usage of SGLang

To demonstrate the agentic capabilities of SGLang, we'll use [qwen-code](https://github.com/QwenLM/qwen-code) as the platform to interact with the SGLang server you just set up.

1. **(Optional)** Run the following command to enter the SGLang container. (Installing qwen-code in the SGLang container allows you to use qwen-code's built-in tools to explore the SGLang repo, such as asking for code explanations.)
    ```bash
    docker exec -it sglang_gpt_oss_server /bin/zsh
    ```

2. Install Node.js first by following this [link](https://nodejs.org/en/download), or run the following commands in the terminal:
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

3. Install qwen-code:
    ```bash
    npm install -g @qwen-code/qwen-code@latest
    qwen --version
    ```
    You should see the qwen-code version displayed.

4. **Important**: Set the endpoint environment variables to point to your SGLang server:
    ```bash
    export OPENAI_API_KEY="sk-123456" # dummy key
    export OPENAI_BASE_URL="http://localhost:30000/v1"
    export OPENAI_MODEL="openai/gpt-oss-20b"
    ```

5. Now you can use qwen-code to interact with the SGLang server. (**Tip**: For better visibility, change the JupyterLab theme to "Dark" via **Settings > Theme > JupyterLab Dark**.)
    ```bash
    qwen
    ```
    For example, type `/init` to let qwen-code analyze the current directory. You'll see several tools being called.
    
    ![qwen-code-init](qwen_code_init.png)

