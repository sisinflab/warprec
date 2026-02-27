# Authentication

In order to interact with the different remote storage services, **WarpRec** requires proper authentication and credential management.

Each remote storage service has its own authentication mechanisms and best practices.

!!! note

    All remote I/O modules in WarpRec are part of the extra *remote-io*. You can install them via pip:

    ```bash
    pip install warprec[remote-io]
    ```

## Azure Blob Storage

The Azure Blob Storage I/O modules in **WarpRec** support various authentication methods to securely connect to Azure services. WarpRec delegates Azure **authentication and credential management** entirely to the user. The I/O modules internally rely on the **Azure Identity** library, which provides secure and flexible authentication mechanisms.

### Supported Authentication Methods

The user can log in to Azure in several ways, including but not limited to:

- **Azure CLI Login:**

  ```bash
  az login
  ```

  WarpRec will automatically detect the active Azure session.

- **Environment Variables:**

  ```bash
  export AZURE_CLIENT_ID=<client_id>
  export AZURE_TENANT_ID=<tenant_id>
  export AZURE_CLIENT_SECRET=<client_secret>
  ```

  These credentials will be used to authenticate non-interactively.

- **Managed Identity (e.g., Azure VM, AKS, App Service):**
  If WarpRec runs in an Azure environment with a managed identity, credentials will be retrieved automatically.

- **Visual Studio Code / Azure Developer Tools Login:**
  If you're logged into Azure through VS Code or Visual Studio, those credentials will also be detected automatically.

### Dynamic Credential Retrieval

WarpRec uses the `DefaultAzureCredential` from the **azure-identity** package to dynamically retrieve credentials. This class transparently checks multiple authentication sources in order of priority and selects the first available one. No manual credential passing is needed -- the process is fully automated and secure.
