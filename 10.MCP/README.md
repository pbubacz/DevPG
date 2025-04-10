# Lab 10: Model Context Protocol

This lab demonstrates a simple example of using the Model Context Protocol (MCP) to interact with a model. The plugin acts as a client that sends requests to the model and displays the responses.

## Inspecting Filesystem Server Commands

To review available commands for the filesystem server, run the following in the terminal:

```bash
npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-filesystem .
```

Then, open your browser and navigate to:

```
http://127.0.0.1:6274
```

Click "Connect" to link to the server and review the available options. Afterward, close the browser and stop the inspector by pressing Ctrl+C in the terminal.

## Running the Client

1. Navigate to the client folder and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the client with:

   ```bash
   chainlit run app.py
   ```

3. Click the plugin icon and, under the "Connect an MCP" tab:
   - Select type: `stdio`
   - Enter **local** in the Name field.
   - Enter **npx -y @modelcontextprotocol/server-filesystem .** in the Command field.
   - Click "Confirm."

## Creating a New MCP Server

1. Navigate to the `server` folder.

2. Install dependencies:

   ```bash
   pip install fastmcp
   ```

3. Start the server:

   ```bash
   fastmcp dev server.py
   ```

4. Add features to the server using the [FastMCP documentation](https://github.com/jlowin/fastmcp).