#include "VulkanImGuiBackend.h"

#include "../../../Engine/GraphicAPI/Vulkan/VulkanContext.h"
#include "../../../Engine/logRecord/logRecordHistory.h"

#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

// Internal symbol from imgui_impl_vulkan.cpp; not exported by the public header (same pattern as DX12 CreateDeviceObjects).
extern bool ImGui_ImplVulkan_CreateDeviceObjects();

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace ImGuiWidgets {

namespace {

// Layout must match ViewportData / Data prefixes in imgui_impl_vulkan.cpp (render side reads handles only; do not touch GImGui).
struct VkFrameRenderBuffers {
    VkDeviceMemory VertexBufferMemory;
    VkDeviceMemory IndexBufferMemory;
    VkDeviceSize VertexBufferSize;
    VkDeviceSize IndexBufferSize;
    VkBuffer VertexBuffer;
    VkBuffer IndexBuffer;
};

struct VkWindowRenderBuffers {
    uint32_t Index;
    uint32_t Count;
    ImVector<VkFrameRenderBuffers> FrameRenderBuffers;
};

struct VkViewportData {
    ImGui_ImplVulkanH_Window Window;
    VkWindowRenderBuffers RenderBuffers;
    bool WindowOwned;
    bool SwapChainNeedRebuild;
    bool SwapChainSuboptimal;
};

struct VkBackendData {
    ImGui_ImplVulkan_InitInfo VulkanInitInfo;
    VkDeviceSize BufferMemoryAlignment;
    VkDeviceSize NonCoherentAtomSize;
    VkPipelineCreateFlags PipelineCreateFlags;
    VkDescriptorSetLayout DescriptorSetLayout;
    VkPipelineLayout PipelineLayout;
    VkPipeline Pipeline;
    VkPipeline PipelineForViewports;
};

inline VkDeviceSize alignBufferSize(VkDeviceSize size, VkDeviceSize alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

inline VkDevice deviceFromContext(const VulkanImGuiBackendContextInfo& contextInfo)
{
    return contextInfo.windowContext ? contextInfo.windowContext->getDevice() : VK_NULL_HANDLE;
}

inline VkPhysicalDevice physicalDeviceFromContext(const VulkanImGuiBackendContextInfo& contextInfo)
{
    return contextInfo.windowContext ? contextInfo.windowContext->getPhysicalDevice()
                                     : VK_NULL_HANDLE;
}

} // namespace

VulkanImGuiBackend::VulkanImGuiBackend(const VulkanImGuiBackendContextInfo& contextInfo)
    : m_contextInfo(contextInfo)
{
}

VulkanImGuiBackend::~VulkanImGuiBackend()
{
    Shutdown();
}

void VulkanImGuiBackend::checkVkResult(VkResult err)
{
    if (err == VK_SUCCESS) {
        return;
    }
    LOG_ERROR_FMT("[ImGuiBackend][Vulkan] Vulkan error: %d", static_cast<int>(err));
}

void VulkanImGuiBackend::waitDeviceIdle()
{
    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    VkDevice const device = windowCtx ? windowCtx->getDevice() : VK_NULL_HANDLE;
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }
}

uint32_t VulkanImGuiBackend::findMemoryType(
    uint32_t typeBits,
    VkMemoryPropertyFlags properties) const
{
    VkPhysicalDevice const physicalDevice = physicalDeviceFromContext(m_contextInfo);
    if (physicalDevice == VK_NULL_HANDLE) {
        return 0;
    }

    VkPhysicalDeviceMemoryProperties prop{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &prop);
    for (uint32_t i = 0; i < prop.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) != 0
            && (prop.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0xFFFFFFFF;
}

bool VulkanImGuiBackend::createOrResizeBuffer(
    VkBuffer& buffer,
    VkDeviceMemory& memory,
    VkDeviceSize& size,
    VkDeviceSize newSize,
    VkBufferUsageFlagBits usage,
    VkDeviceSize bufferMemoryAlignment)
{
    VkDevice const device = deviceFromContext(m_contextInfo);
    if (device == VK_NULL_HANDLE) {
        return false;
    }

    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }

    constexpr VkDeviceSize kMinAlloc = 1024;
    const VkDeviceSize alignment = bufferMemoryAlignment > 0 ? bufferMemoryAlignment : 256;
    const VkDeviceSize bufferSizeAligned =
        alignBufferSize((std::max)(kMinAlloc, newSize), alignment);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSizeAligned;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] createOrResizeBuffer: vkCreateBuffer failed");
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buffer, &req);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (allocInfo.memoryTypeIndex == 0xFFFFFFFF
        || vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] createOrResizeBuffer: vkAllocateMemory failed");
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] createOrResizeBuffer: vkBindBufferMemory failed");
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }

    size = bufferSizeAligned;
    return true;
}

bool VulkanImGuiBackend::ensureOfficialDeviceObjects()
{
    if (!ImGui_ImplVulkan_CreateDeviceObjects()) {
        LOG_ERROR("[ImGuiBackend][Vulkan] ImGui_ImplVulkan_CreateDeviceObjects failed");
        return false;
    }
    return true;
}

bool VulkanImGuiBackend::createSubmitSlots()
{
    destroySubmitSlots();

    if (deviceFromContext(m_contextInfo) == VK_NULL_HANDLE) {
        return false;
    }

    VkDeviceSize alignment = 256;
    auto* const bd = static_cast<VkBackendData*>(ImGui::GetIO().BackendRendererUserData);
    if (bd && bd->BufferMemoryAlignment > 0) {
        alignment = bd->BufferMemoryAlignment;
    }

    for (VulkanImGuiSubmitSlot& slot : m_submitSlots) {
        const VkDeviceSize vbBytes =
            alignBufferSize(5000 * sizeof(ImDrawVert), alignment);
        const VkDeviceSize ibBytes =
            alignBufferSize(10000 * sizeof(ImDrawIdx), alignment);
        if (!createOrResizeBuffer(
                slot.vb,
                slot.vbMemory,
                slot.vbSize,
                vbBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                alignment)
            || !createOrResizeBuffer(
                slot.ib,
                slot.ibMemory,
                slot.ibSize,
                ibBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                alignment)) {
            LOG_ERROR("[ImGuiBackend][Vulkan] createSubmitSlots failed");
            destroySubmitSlots();
            return false;
        }
    }

    m_recordSlotIndex = 0;
    return true;
}

void VulkanImGuiBackend::destroySubmitSlots()
{
    VkDevice const device = deviceFromContext(m_contextInfo);
    for (VulkanImGuiSubmitSlot& slot : m_submitSlots) {
        if (device != VK_NULL_HANDLE) {
            if (slot.vb != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, slot.vb, nullptr);
            }
            if (slot.vbMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, slot.vbMemory, nullptr);
            }
            if (slot.ib != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, slot.ib, nullptr);
            }
            if (slot.ibMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, slot.ibMemory, nullptr);
            }
        }
        slot = {};
    }
}

bool VulkanImGuiBackend::Init(const bool installCallbacks)
{
    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    if (windowCtx == nullptr
        || windowCtx->getInstance() == VK_NULL_HANDLE
        || windowCtx->getDevice() == VK_NULL_HANDLE
        || windowCtx->getDescriptorPool() == VK_NULL_HANDLE
        || windowCtx->getGraphicsQueue() == VK_NULL_HANDLE
        || windowCtx->getRenderPass() == VK_NULL_HANDLE) {
        LOG_ERROR(
            "[ImGuiBackend][Vulkan] windowContext / instance / device / "
            "descriptorPool / queue / renderPass is null");
        return false;
    }

    GLFWwindow* const window = windowCtx->getWindow();
    if (window == nullptr) {
        LOG_ERROR("[ImGuiBackend][Vulkan] windowContext has no GLFW window");
        return false;
    }

    m_imguiContext = ConfigureCommonImGuiContext();

    if (!initPlatformAndRenderer(installCallbacks)) {
        DestroyImGuiBackendContext(m_imguiContext);
        m_imguiContext = nullptr;
        return false;
    }

    if (!createOffscreenTextureSampler()) {
        shutdownPlatformAndRenderer();
        DestroyImGuiBackendContext(m_imguiContext);
        m_imguiContext = nullptr;
        return false;
    }

    ImGui::SetCurrentContext(m_imguiContext);
    if (!ensureOfficialDeviceObjects() || !createSubmitSlots()) {
        LOG_ERROR("[ImGuiBackend][Vulkan] submit slot / pipeline init failed");
        destroySubmitSlots();
        shutdownPlatformAndRenderer();
        DestroyImGuiBackendContext(m_imguiContext);
        m_imguiContext = nullptr;
        return false;
    }

    installRendererViewportHooks();
    //installPlatformViewportHooks();

    m_drawDataCache.configFlags = ImGui::GetIO().ConfigFlags;

    LOG_INFO(
        "[ImGuiBackend][Vulkan] initialized "
        "(jobs queue + render-thread self draw)");
    return true;
}

bool VulkanImGuiBackend::initPlatformAndRenderer(const bool installCallbacks)
{
    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    if (windowCtx == nullptr || m_imguiContext == nullptr) {
        LOG_ERROR("[ImGuiBackend][Vulkan] initPlatformAndRenderer: context not ready");
        return false;
    }

    GLFWwindow* const window = windowCtx->getWindow();
    if (window == nullptr) {
        LOG_ERROR("[ImGuiBackend][Vulkan] initPlatformAndRenderer: GLFW window is null");
        return false;
    }

    ImGui::SetCurrentContext(m_imguiContext);

    if (!ImGui_ImplGlfw_InitForVulkan(window, installCallbacks)) {
        LOG_ERROR("[ImGuiBackend][Vulkan] ImGui_ImplGlfw_InitForVulkan failed");
        return false;
    }

    if (!createVulkanBackendObjects()) {
        ImGui_ImplGlfw_Shutdown();
        return false;
    }

    return true;
}

void VulkanImGuiBackend::shutdownPlatformAndRenderer()
{
    if (m_imguiContext == nullptr) {
        return;
    }

    ImGui::SetCurrentContext(m_imguiContext);
    const bool hasPlatform = ImGui::GetIO().BackendPlatformUserData != nullptr;
    destroyVulkanBackendObjects();
    if (hasPlatform) {
        ImGui_ImplGlfw_Shutdown();
    }
}

bool VulkanImGuiBackend::createVulkanBackendObjects()
{
    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    if (windowCtx == nullptr
        || windowCtx->getInstance() == VK_NULL_HANDLE
        || windowCtx->getDevice() == VK_NULL_HANDLE
        || windowCtx->getDescriptorPool() == VK_NULL_HANDLE
        || windowCtx->getGraphicsQueue() == VK_NULL_HANDLE
        || windowCtx->getRenderPass() == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] createVulkanBackendObjects: invalid Vulkan handles");
        return false;
    }

    ImGui::SetCurrentContext(m_imguiContext);

    m_minImageCount = (std::max)(2u, windowCtx->getSwapChainImageCount());

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = windowCtx->getInstance();
    initInfo.PhysicalDevice = windowCtx->getPhysicalDevice();
    initInfo.Device = windowCtx->getDevice();
    initInfo.QueueFamily = windowCtx->getGraphicsQueueFamily();
    initInfo.Queue = windowCtx->getGraphicsQueue();
    initInfo.PipelineCache = windowCtx->getPipelineCache();
    initInfo.DescriptorPool = windowCtx->getDescriptorPool();
    initInfo.MinImageCount = m_minImageCount;
    initInfo.ImageCount = m_minImageCount;
    initInfo.Allocator = nullptr;
    initInfo.PipelineInfoMain.RenderPass = windowCtx->getRenderPass();
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CheckVkResultFn = &VulkanImGuiBackend::checkVkResult;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        LOG_ERROR("[ImGuiBackend][Vulkan] ImGui_ImplVulkan_Init failed");
        return false;
    }

    return true;
}

void VulkanImGuiBackend::destroyVulkanBackendObjects()
{
    if (m_imguiContext == nullptr) {
        return;
    }

    ImGui::SetCurrentContext(m_imguiContext);
    if (ImGui::GetIO().BackendRendererUserData == nullptr) {
        return;
    }

    waitDeviceIdle();
    destroySubmitSlots();
    destroyTextureUploadCommandResources();
    ImGui_ImplVulkan_Shutdown();
    m_drawData = nullptr;
}

void VulkanImGuiBackend::flushPendingTexturesOnEventThread()
{
    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    for (ImTextureData* tex : pio.Textures) {
        if (tex && tex->Status != ImTextureStatus_OK) {
            ImGui_ImplVulkan_UpdateTexture(tex);
        }
    }
}

void VulkanImGuiBackend::destroyTextureUploadStagingResources(
    std::vector<VulkanTexUploadStaging>& stagings)
{
    VkDevice const device = deviceFromContext(m_contextInfo);
    if (device != VK_NULL_HANDLE) {
        for (VulkanTexUploadStaging& staging : stagings) {
            if (staging.buffer) {
                vkDestroyBuffer(device, static_cast<VkBuffer>(staging.buffer), nullptr);
            }
            if (staging.memory) {
                vkFreeMemory(device, static_cast<VkDeviceMemory>(staging.memory), nullptr);
            }
            staging = {};
        }
    }
    stagings.clear();
}

bool VulkanImGuiBackend::ensureTextureUploadCommandResources()
{
    if (m_texUploadCommandPool != VK_NULL_HANDLE && m_texUploadCommandBuffer != VK_NULL_HANDLE) {
        return true;
    }

    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    VkDevice const device = windowCtx ? windowCtx->getDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) {
        return false;
    }

    // If half-initialized, destroy and recreate so pool/CB stay paired.
    destroyTextureUploadCommandResources();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.queueFamilyIndex = windowCtx->getGraphicsQueueFamily();
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_texUploadCommandPool) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] ensureTextureUpload: CommandPool failed");
        m_texUploadCommandPool = VK_NULL_HANDLE;
        return false;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_texUploadCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &allocInfo, &m_texUploadCommandBuffer) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] ensureTextureUpload: CommandBuffer failed");
        vkDestroyCommandPool(device, m_texUploadCommandPool, nullptr);
        m_texUploadCommandPool = VK_NULL_HANDLE;
        m_texUploadCommandBuffer = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void VulkanImGuiBackend::destroyTextureUploadCommandResources()
{
    VkDevice const device = deviceFromContext(m_contextInfo);
    if (device != VK_NULL_HANDLE && m_texUploadCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, m_texUploadCommandPool, nullptr);
    }
    m_texUploadCommandPool = VK_NULL_HANDLE;
    m_texUploadCommandBuffer = VK_NULL_HANDLE;
}

void VulkanImGuiBackend::enrichSecondaryViewportJob(
    SecondaryViewportJob& job,
    ImGuiViewport* viewport)
{
    (void)viewport;
    // Event thread still owns GImGui: snapshot official bd pipelines into this job; render side reads the snapshot only.
    // Secondary PipelineForViewports may become non-null only after CreateWindow; capturing at enrich this frame is enough.
    auto* const bd = static_cast<VkBackendData*>(ImGui::GetIO().BackendRendererUserData);
    if (!bd) {
        return;
    }
    if (bd->Pipeline != VK_NULL_HANDLE) {
        job.vulkanPipeline = bd->Pipeline;
    }
    if (bd->PipelineForViewports != VK_NULL_HANDLE) {
        job.vulkanPipelineForViewports = bd->PipelineForViewports;
    }
    if (bd->PipelineLayout != VK_NULL_HANDLE) {
        job.vulkanPipelineLayout = bd->PipelineLayout;
    }
    if (bd->BufferMemoryAlignment > 0) {
        job.vulkanBufferMemoryAlignment = bd->BufferMemoryAlignment;
    }
}

void VulkanImGuiBackend::releaseSecondaryViewportJobGraphicsResources(SecondaryViewportJob& job)
{
    // Pool/CB are long-lived class members; only free this frame's staging and clear the submit flag.
    destroyTextureUploadStagingResources(job.vulkanTexUploadStagings);
    job.vulkanSubmitTextureUploads = false;
}

void VulkanImGuiBackend::markSecondaryViewportSwapChainNeedRebuildForRender(ImGuiViewport* viewport)
{
    // Event-side Resize skipped official SetWindowSize; mark NeedRebuild for our pre-draw recreate path.
    if (!viewport || !viewport->RendererUserData) {
        return;
    }
    auto* const vd = static_cast<VkViewportData*>(viewport->RendererUserData);
    vd->SwapChainNeedRebuild = true;
}

namespace {

void destroySecondaryFrame(VkDevice device, ImGui_ImplVulkanH_Frame& fd)
{
    if (fd.Fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, fd.Fence, nullptr);
        fd.Fence = VK_NULL_HANDLE;
    }
    if (fd.CommandPool != VK_NULL_HANDLE) {
        if (fd.CommandBuffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, fd.CommandPool, 1, &fd.CommandBuffer);
            fd.CommandBuffer = VK_NULL_HANDLE;
        }
        vkDestroyCommandPool(device, fd.CommandPool, nullptr);
        fd.CommandPool = VK_NULL_HANDLE;
    }
    if (fd.BackbufferView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, fd.BackbufferView, nullptr);
        fd.BackbufferView = VK_NULL_HANDLE;
    }
    if (fd.Framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, fd.Framebuffer, nullptr);
        fd.Framebuffer = VK_NULL_HANDLE;
    }
    fd.Backbuffer = VK_NULL_HANDLE;
}

void destroySecondaryFrameSemaphores(VkDevice device, ImGui_ImplVulkanH_FrameSemaphores& fsd)
{
    if (fsd.ImageAcquiredSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, fsd.ImageAcquiredSemaphore, nullptr);
        fsd.ImageAcquiredSemaphore = VK_NULL_HANDLE;
    }
    if (fsd.RenderCompleteSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, fsd.RenderCompleteSemaphore, nullptr);
        fsd.RenderCompleteSemaphore = VK_NULL_HANDLE;
    }
}

} // namespace

bool VulkanImGuiBackend::recreateSecondaryViewportSwapchain(
    ImGui_ImplVulkanH_Window* wd,
    int width,
    int height)
{
    // Same idea as main-window recreatePresentationResources: transition via oldSwapchain; sync only this window's submits.
    // Avoid official CreateOrResize QueueWaitIdle + temporary layout Submit (easy to widen hang surface while dragging).
    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    if (!wd || !windowCtx || width <= 0 || height <= 0) {
        return false;
    }
    if (wd->Surface == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: Surface null");
        return false;
    }

    VkDevice const device = windowCtx->getDevice();
    VkPhysicalDevice const physicalDevice = windowCtx->getPhysicalDevice();
    VkQueue const queue = windowCtx->getGraphicsQueue();
    const uint32_t graphicsFamily = windowCtx->getGraphicsQueueFamily();
    const uint32_t presentFamily = windowCtx->getPresentQueueFamily();
    if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) {
        return false;
    }

    // 1) Wait this window's per-image fences (Submit done); then empty Submit+Fence to drain unfinished Present on the same queue.
    // 5s timeout: avoid infinite block on a bad queue; on failure keep NeedRebuild and retry next frame.
    constexpr std::uint64_t kRecreateWaitTimeoutNs = 5000ull * 1000ull * 1000ull;
    for (int i = 0; i < wd->Frames.Size; ++i) {
        VkFence const fence = wd->Frames[i].Fence;
        if (fence == VK_NULL_HANDLE) {
            continue;
        }
        const VkResult waitErr =
            vkWaitForFences(device, 1, &fence, VK_TRUE, kRecreateWaitTimeoutNs);
        if (waitErr == VK_TIMEOUT) {
            LOG_WARNING_FMT(
                "[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: WaitForFences[%d] timed out",
                i);
            return false;
        }
        if (waitErr != VK_SUCCESS) {
            LOG_ERROR_FMT(
                "[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: WaitForFences[%d] failed (%d)",
                i,
                static_cast<int>(waitErr));
            return false;
        }
    }
    {
        VkFenceCreateInfo drainFenceInfo{};
        drainFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence drainFence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &drainFenceInfo, nullptr, &drainFence) != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateFence(drain) failed");
            return false;
        }
        const VkResult submitErr = vkQueueSubmit(queue, 0, nullptr, drainFence);
        if (submitErr != VK_SUCCESS) {
            LOG_ERROR_FMT(
                "[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: drain QueueSubmit failed (%d)",
                static_cast<int>(submitErr));
            vkDestroyFence(device, drainFence, nullptr);
            return false;
        }
        const VkResult drainWait =
            vkWaitForFences(device, 1, &drainFence, VK_TRUE, kRecreateWaitTimeoutNs);
        if (drainWait != VK_SUCCESS) {
            LOG_ERROR_FMT(
                "[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: drain WaitForFences failed (%d)",
                static_cast<int>(drainWait));
            // On TIMEOUT drainFence is still pending; intentionally do not destroy it.
            if (drainWait != VK_TIMEOUT) {
                vkDestroyFence(device, drainFence, nullptr);
            }
            return false;
        }
        vkDestroyFence(device, drainFence, nullptr);
    }

    // 2) Tear down old frame resources (keep Surface / SurfaceFormat / PresentMode / AttachmentDesc).
    const VkSwapchainKHR oldSwapchain = wd->Swapchain;
    wd->Swapchain = VK_NULL_HANDLE;
    for (int i = 0; i < wd->Frames.Size; ++i) {
        destroySecondaryFrame(device, wd->Frames[i]);
    }
    for (int i = 0; i < wd->FrameSemaphores.Size; ++i) {
        destroySecondaryFrameSemaphores(device, wd->FrameSemaphores[i]);
    }
    wd->Frames.clear();
    wd->FrameSemaphores.clear();
    wd->ImageCount = 0;
    wd->SemaphoreCount = 0;
    wd->FrameIndex = 0;
    wd->SemaphoreIndex = 0;
    if (wd->RenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, wd->RenderPass, nullptr);
        wd->RenderPass = VK_NULL_HANDLE;
    }

    uint32_t minImageCount = m_minImageCount;
    if (minImageCount < 2) {
        minImageCount = static_cast<uint32_t>(
            ImGui_ImplVulkanH_GetMinImageCountFromPresentMode(wd->PresentMode));
    }

    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, wd->Surface, &caps)
        != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: getSurfaceCapabilities failed");
        if (oldSwapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
        }
        return false;
    }

    uint32_t imageCount = minImageCount;
    if (imageCount < caps.minImageCount) {
        imageCount = caps.minImageCount;
    }
    if (caps.maxImageCount != 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkExtent2D extent{};
    if (caps.currentExtent.width == 0xffffffffu) {
        extent.width = static_cast<uint32_t>(width);
        extent.height = static_cast<uint32_t>(height);
    } else {
        extent = caps.currentExtent;
    }
    if (extent.width == 0 || extent.height == 0) {
        LOG_WARNING("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: extent 0, skip");
        if (oldSwapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
        }
        return false;
    }
    wd->Width = static_cast<int>(extent.width);
    wd->Height = static_cast<int>(extent.height);

    VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
        composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
        composite = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
        composite = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
        composite = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
    }

    VkSwapchainCreateInfoKHR swapInfo{};
    swapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapInfo.surface = wd->Surface;
    swapInfo.minImageCount = imageCount;
    swapInfo.imageFormat = wd->SurfaceFormat.format;
    swapInfo.imageColorSpace = wd->SurfaceFormat.colorSpace;
    swapInfo.imageExtent = extent;
    swapInfo.imageArrayLayers = 1;
    swapInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const uint32_t queueFamilies[] = { graphicsFamily, presentFamily };
    if (graphicsFamily != presentFamily) {
        swapInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapInfo.queueFamilyIndexCount = 2;
        swapInfo.pQueueFamilyIndices = queueFamilies;
    } else {
        swapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    swapInfo.preTransform =
        (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
            : caps.currentTransform;
    swapInfo.compositeAlpha = composite;
    swapInfo.presentMode = wd->PresentMode;
    swapInfo.clipped = VK_TRUE;
    swapInfo.oldSwapchain = oldSwapchain;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device, &swapInfo, nullptr, &newSwapchain) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateSwapchain failed");
        if (oldSwapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
        }
        return false;
    }
    if (oldSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
    }
    wd->Swapchain = newSwapchain;

    uint32_t swapImageCount = 0;
    vkGetSwapchainImagesKHR(device, wd->Swapchain, &swapImageCount, nullptr);
    if (swapImageCount == 0 || swapImageCount > 16) {
        LOG_ERROR_FMT(
            "[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: bad ImageCount=%u",
            swapImageCount);
        return false;
    }
    VkImage backbuffers[16]{};
    vkGetSwapchainImagesKHR(device, wd->Swapchain, &swapImageCount, backbuffers);
    wd->ImageCount = swapImageCount;
    wd->SemaphoreCount = wd->ImageCount + 1;
    wd->Frames.resize(static_cast<int>(wd->ImageCount));
    wd->FrameSemaphores.resize(static_cast<int>(wd->SemaphoreCount));
    std::memset(wd->Frames.Data, 0, static_cast<size_t>(wd->Frames.size_in_bytes()));
    std::memset(
        wd->FrameSemaphores.Data,
        0,
        static_cast<size_t>(wd->FrameSemaphores.size_in_bytes()));
    for (uint32_t i = 0; i < wd->ImageCount; ++i) {
        wd->Frames[static_cast<int>(i)].Backbuffer = backbuffers[i];
    }

    // RenderPass (aligned with official AttachmentDesc / secondary clear conventions)
    if (!wd->UseDynamicRendering) {
        VkAttachmentDescription attachment = wd->AttachmentDesc;
        if (attachment.format == VK_FORMAT_UNDEFINED) {
            attachment.format = wd->SurfaceFormat.format;
        }
        VkAttachmentReference colorAttachment{};
        colorAttachment.attachment = 0;
        colorAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachment;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &attachment;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dependency;
        if (vkCreateRenderPass(device, &rpInfo, nullptr, &wd->RenderPass) != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateRenderPass failed");
            return false;
        }
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = wd->SurfaceFormat.format;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    for (uint32_t i = 0; i < wd->ImageCount; ++i) {
        ImGui_ImplVulkanH_Frame& fd = wd->Frames[static_cast<int>(i)];
        viewInfo.image = fd.Backbuffer;
        if (vkCreateImageView(device, &viewInfo, nullptr, &fd.BackbufferView) != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateImageView failed");
            return false;
        }
    }

    if (!wd->UseDynamicRendering && wd->RenderPass != VK_NULL_HANDLE) {
        VkImageView attachmentViews[1]{};
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = wd->RenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = attachmentViews;
        fbInfo.width = static_cast<uint32_t>(wd->Width);
        fbInfo.height = static_cast<uint32_t>(wd->Height);
        fbInfo.layers = 1;
        for (uint32_t i = 0; i < wd->ImageCount; ++i) {
            ImGui_ImplVulkanH_Frame& fd = wd->Frames[static_cast<int>(i)];
            attachmentViews[0] = fd.BackbufferView;
            if (vkCreateFramebuffer(device, &fbInfo, nullptr, &fd.Framebuffer) != VK_SUCCESS) {
                LOG_ERROR(
                    "[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateFramebuffer failed");
                return false;
            }
        }
    }

    // Per image: CommandPool / CB / Fence(signaled); semaphore count = ImageCount+1 (safe Acquire reuse).
    for (uint32_t i = 0; i < wd->ImageCount; ++i) {
        ImGui_ImplVulkanH_Frame& fd = wd->Frames[static_cast<int>(i)];
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = graphicsFamily;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &fd.CommandPool) != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateCommandPool failed");
            return false;
        }
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = fd.CommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocInfo, &fd.CommandBuffer) != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: AllocateCB failed");
            return false;
        }
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fd.Fence) != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateFence failed");
            return false;
        }
    }
    for (uint32_t i = 0; i < wd->SemaphoreCount; ++i) {
        ImGui_ImplVulkanH_FrameSemaphores& fsd = wd->FrameSemaphores[static_cast<int>(i)];
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semInfo, nullptr, &fsd.ImageAcquiredSemaphore) != VK_SUCCESS
            || vkCreateSemaphore(device, &semInfo, nullptr, &fsd.RenderCompleteSemaphore)
                != VK_SUCCESS) {
            LOG_ERROR("[ImGuiBackend][Vulkan] recreateSecondaryViewportSwapchain: CreateSemaphore failed");
            return false;
        }
    }

    // Skip official layout->PRESENT temp Submit: RP initialLayout=UNDEFINED, first-frame BeginRP is enough.
    return true;
}

bool VulkanImGuiBackend::beginTextureUploadRecording(
    VkCommandPool commandPool,
    VkCommandBuffer commandBuffer)
{
    if (commandBuffer == VK_NULL_HANDLE || commandPool == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] beginTextureUploadRecording: invalid pool/CB");
        return false;
    }

    VkDevice const device = deviceFromContext(m_contextInfo);
    if (device == VK_NULL_HANDLE) {
        return false;
    }

    if (vkResetCommandPool(device, commandPool, 0) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] beginTextureUploadRecording: ResetCommandPool failed");
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] beginTextureUploadRecording: BeginCommandBuffer failed");
        return false;
    }
    return true;
}

void VulkanImGuiBackend::appendTextureUploadCopyCommands(
    VkCommandBuffer commandBuffer,
    VkBuffer uploadBuffer,
    VkDeviceSize uploadSize,
    VkImage dstImage,
    int32_t uploadX,
    int32_t uploadY,
    uint32_t uploadW,
    uint32_t uploadH,
    bool fromUndefinedLayout)
{
    if (commandBuffer == VK_NULL_HANDLE || uploadBuffer == VK_NULL_HANDLE
        || dstImage == VK_NULL_HANDLE || uploadW == 0 || uploadH == 0 || uploadSize == 0) {
        return;
    }

    VkBufferMemoryBarrier uploadBarrier{};
    uploadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    uploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    uploadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    uploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uploadBarrier.buffer = uploadBuffer;
    uploadBarrier.offset = 0;
    uploadBarrier.size = uploadSize;

    VkImageMemoryBarrier copyBarrier{};
    copyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    copyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copyBarrier.oldLayout = fromUndefinedLayout ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    copyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copyBarrier.image = dstImage;
    copyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyBarrier.subresourceRange.levelCount = 1;
    copyBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &uploadBarrier,
        1,
        &copyBarrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = uploadW;
    region.imageExtent.height = uploadH;
    region.imageExtent.depth = 1;
    region.imageOffset.x = uploadX;
    region.imageOffset.y = uploadY;
    vkCmdCopyBufferToImage(
        commandBuffer,
        uploadBuffer,
        dstImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    VkImageMemoryBarrier useBarrier{};
    useBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    useBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    useBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    useBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    useBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    useBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    useBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    useBarrier.image = dstImage;
    useBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    useBarrier.subresourceRange.levelCount = 1;
    useBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &useBarrier);
}

bool VulkanImGuiBackend::recordPendingTextureUploadsOnEventPhase(
    std::vector<VulkanTexUploadStaging>& outStagings)
{
    // Layout matches imgui_impl_vulkan.cpp::ImGui_ImplVulkan_Texture (BackendUserData).
    struct VkImGuiTexture {
        VkDeviceMemory Memory = VK_NULL_HANDLE;
        VkImage Image = VK_NULL_HANDLE;
        VkImageView ImageView = VK_NULL_HANDLE;
        VkDescriptorSet DescriptorSet = VK_NULL_HANDLE;
    };

    outStagings.clear();

    VkDevice const device = deviceFromContext(m_contextInfo);
    if (device == VK_NULL_HANDLE) {
        return false;
    }

    auto* const bd = static_cast<VkBackendData*>(ImGui::GetIO().BackendRendererUserData);
    if (!bd) {
        return false;
    }

    // AddTexture uses the offscreen sampler (same as registerOffscreenTexture); official UpdateTexture uses TexSamplerLinear.
    VkSampler const sampler = m_offscreenTextureSampler;
    if (sampler == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] recordPendingTextureUploads: sampler null");
        return false;
    }

    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    bool began = false;
    const VkDeviceSize minAlloc = bd->VulkanInitInfo.MinAllocationSize > 0
        ? bd->VulkanInitInfo.MinAllocationSize
        : 1024;
    const VkDeviceSize atom = bd->NonCoherentAtomSize > 0 ? bd->NonCoherentAtomSize : 64;

    for (ImTextureData* tex : pio.Textures) {
        if (!tex
            || (tex->Status != ImTextureStatus_WantCreate
                && tex->Status != ImTextureStatus_WantUpdates)) {
            continue;
        }
        if (tex->Format != ImTextureFormat_RGBA32 || !tex->Pixels || tex->Width <= 0
            || tex->Height <= 0) {
            continue;
        }

        const bool isCreate = (tex->Status == ImTextureStatus_WantCreate);
        VkImGuiTexture* backendTex = static_cast<VkImGuiTexture*>(tex->BackendUserData);

        if (isCreate) {
            if (tex->TexID != ImTextureID_Invalid || backendTex != nullptr) {
                continue;
            }
            backendTex = IM_NEW(VkImGuiTexture)();

            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageInfo.extent.width = static_cast<uint32_t>(tex->Width);
            imageInfo.extent.height = static_cast<uint32_t>(tex->Height);
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (vkCreateImage(device, &imageInfo, nullptr, &backendTex->Image) != VK_SUCCESS) {
                IM_DELETE(backendTex);
                LOG_ERROR("[ImGuiBackend][Vulkan] recordPendingTextureUploads: CreateImage failed");
                continue;
            }

            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(device, backendTex->Image, &req);
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = (std::max)(minAlloc, req.size);
            allocInfo.memoryTypeIndex =
                findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (allocInfo.memoryTypeIndex == 0xFFFFFFFF
                || vkAllocateMemory(device, &allocInfo, nullptr, &backendTex->Memory)
                    != VK_SUCCESS) {
                vkDestroyImage(device, backendTex->Image, nullptr);
                IM_DELETE(backendTex);
                continue;
            }
            vkBindImageMemory(device, backendTex->Image, backendTex->Memory, 0);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = backendTex->Image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &viewInfo, nullptr, &backendTex->ImageView)
                != VK_SUCCESS) {
                vkFreeMemory(device, backendTex->Memory, nullptr);
                vkDestroyImage(device, backendTex->Image, nullptr);
                IM_DELETE(backendTex);
                continue;
            }

            backendTex->DescriptorSet = ImGui_ImplVulkan_AddTexture(
                sampler, backendTex->ImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (backendTex->DescriptorSet == VK_NULL_HANDLE) {
                vkDestroyImageView(device, backendTex->ImageView, nullptr);
                vkFreeMemory(device, backendTex->Memory, nullptr);
                vkDestroyImage(device, backendTex->Image, nullptr);
                IM_DELETE(backendTex);
                continue;
            }

            tex->SetTexID(reinterpret_cast<ImTextureID>(backendTex->DescriptorSet));
            tex->BackendUserData = backendTex;
        } else if (!backendTex || backendTex->Image == VK_NULL_HANDLE) {
            continue;
        }

        const int uploadX = isCreate ? 0 : tex->UpdateRect.x;
        const int uploadY = isCreate ? 0 : tex->UpdateRect.y;
        const int uploadW = isCreate ? tex->Width : tex->UpdateRect.w;
        const int uploadH = isCreate ? tex->Height : tex->UpdateRect.h;
        if (uploadW <= 0 || uploadH <= 0) {
            continue;
        }

        const VkDeviceSize uploadPitch =
            static_cast<VkDeviceSize>(uploadW) * static_cast<VkDeviceSize>(tex->BytesPerPixel);
        const VkDeviceSize uploadSize = alignBufferSize(
            static_cast<VkDeviceSize>(uploadH) * uploadPitch, atom);

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = uploadSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
            continue;
        }
        VkMemoryRequirements bufReq{};
        vkGetBufferMemoryRequirements(device, stagingBuffer, &bufReq);
        VkMemoryAllocateInfo bufAlloc{};
        bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        bufAlloc.allocationSize = (std::max)(minAlloc, bufReq.size);
        bufAlloc.memoryTypeIndex =
            findMemoryType(bufReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (bufAlloc.memoryTypeIndex == 0xFFFFFFFF
            || vkAllocateMemory(device, &bufAlloc, nullptr, &stagingMemory) != VK_SUCCESS) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            continue;
        }
        vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

        char* map = nullptr;
        if (vkMapMemory(device, stagingMemory, 0, uploadSize, 0, reinterpret_cast<void**>(&map))
            != VK_SUCCESS) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            continue;
        }
        for (int y = 0; y < uploadH; ++y) {
            std::memcpy(
                map + uploadPitch * static_cast<VkDeviceSize>(y),
                tex->GetPixelsAt(uploadX, uploadY + y),
                static_cast<size_t>(uploadPitch));
        }
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = stagingMemory;
        range.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device, 1, &range);
        vkUnmapMemory(device, stagingMemory);

        if (!began) {
            // Member pool/CB: create once; each frame Reset+Begin then record; render side Submits.
            if (!ensureTextureUploadCommandResources()
                || !beginTextureUploadRecording(
                    m_texUploadCommandPool, m_texUploadCommandBuffer)) {
                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingMemory, nullptr);
                destroyTextureUploadStagingResources(outStagings);
                return false;
            }
            began = true;
        }

        appendTextureUploadCopyCommands(
            m_texUploadCommandBuffer,
            stagingBuffer,
            uploadSize,
            backendTex->Image,
            uploadX,
            uploadY,
            static_cast<uint32_t>(uploadW),
            static_cast<uint32_t>(uploadH),
            isCreate);

        VulkanTexUploadStaging staging{};
        staging.buffer = stagingBuffer;
        staging.memory = stagingMemory;
        outStagings.push_back(staging);
        // Must set OK on the event thread immediately: if TexID/BackendUserData are set but status stays WantCreate,
        // the next NewFrame trips ImGui diagnostic asserts.
        //
        // [Research / not implemented] If jobs are discarded before GPU Submit while status is already OK -> missing glyphs.
        // Official API: ImTextureData::SetStatus(WantCreate|WantUpdates) (use SetStatus; do not write Status directly).
        // - WantUpdates: refill pixels when atlas already has TexID; UpdateRect/Updates[] may already be cleared -- must refill rects yourself.
        // - WantCreate: SetStatus(Destroyed) with Pixels still present may auto-flip back to WantCreate; or set WantCreate directly.
        // - Risk: races with render sampling; WantCreate + existing TexID asserts easily; when unknown which tex were recorded but not submitted,
        //   blind marking may need multiple frames -- high uncertainty; leave state machine unchanged for now.
        tex->SetStatus(ImTextureStatus_OK);
    }

    if (!began) {
        return false;
    }

    return true;
}

void VulkanImGuiBackend::submitPendingTextureUploadsOnRenderPhase(SecondaryViewportJob& mainJob)
{
    // If already submitted (return-jobs path clears the flag), skip End/Submit to avoid re-submitting the shared member CB racing the event side.
    if (!mainJob.vulkanSubmitTextureUploads) {
        return;
    }

    const VkCommandBuffer commandBuffer = m_texUploadCommandBuffer;

    if (mainJob.vulkanTexUploadStagings.empty() || commandBuffer == VK_NULL_HANDLE) {
        releaseSecondaryViewportJobGraphicsResources(mainJob);
        return;
    }

    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    VkQueue const queue = windowCtx ? windowCtx->getGraphicsQueue() : VK_NULL_HANDLE;
    if (queue == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] submitPendingTextureUploads: queue null");
        releaseSecondaryViewportJobGraphicsResources(mainJob);
        return;
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] submitPendingTextureUploads: EndCommandBuffer failed");
        releaseSecondaryViewportJobGraphicsResources(mainJob);
        return;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] submitPendingTextureUploads: QueueSubmit failed");
        releaseSecondaryViewportJobGraphicsResources(mainJob);
        return;
    }

    vkQueueWaitIdle(queue);

    // After Submit: free this frame's staging + clear flag; keep member pool/CB alive.
    releaseSecondaryViewportJobGraphicsResources(mainJob);
}

void VulkanImGuiBackend::setupRenderStateWithSlot(
    VkCommandBuffer commandBuffer,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VulkanImGuiSubmitSlot& slot,
    const ImDrawData* drawData,
    int fbWidth,
    int fbHeight)
{
    if (!commandBuffer || !drawData || pipeline == VK_NULL_HANDLE
        || pipelineLayout == VK_NULL_HANDLE) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    if (drawData->TotalVtxCount > 0 && slot.vb != VK_NULL_HANDLE && slot.ib != VK_NULL_HANDLE) {
        VkBuffer vertexBuffers[1] = { slot.vb };
        VkDeviceSize vertexOffset[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffset);
        vkCmdBindIndexBuffer(
            commandBuffer,
            slot.ib,
            0,
            sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(fbWidth);
    viewport.height = static_cast<float>(fbHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    float constants[4];
    constants[0] = 2.0f / drawData->DisplaySize.x;
    constants[1] = 2.0f / drawData->DisplaySize.y;
    constants[2] = -1.0f - drawData->DisplayPos.x * constants[0];
    constants[3] = -1.0f - drawData->DisplayPos.y * constants[1];
    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(float) * 4,
        constants);
}

void VulkanImGuiBackend::renderDrawDataWithOfficialFrameBuffers(
    VkCommandBuffer commandBuffer,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDeviceSize bufferMemoryAlignment,
    VkDeviceMemory& vertexBufferMemory,
    VkDeviceMemory& indexBufferMemory,
    VkDeviceSize& vertexBufferSize,
    VkDeviceSize& indexBufferSize,
    VkBuffer& vertexBuffer,
    VkBuffer& indexBuffer,
    ImDrawData* drawData)
{
    VulkanImGuiSubmitSlot slot{};
    slot.vb = vertexBuffer;
    slot.vbMemory = vertexBufferMemory;
    slot.vbSize = vertexBufferSize;
    slot.ib = indexBuffer;
    slot.ibMemory = indexBufferMemory;
    slot.ibSize = indexBufferSize;
    renderDrawDataWithSlot(
        commandBuffer, pipeline, pipelineLayout, bufferMemoryAlignment, slot, drawData);
    vertexBuffer = slot.vb;
    vertexBufferMemory = slot.vbMemory;
    vertexBufferSize = slot.vbSize;
    indexBuffer = slot.ib;
    indexBufferMemory = slot.ibMemory;
    indexBufferSize = slot.ibSize;
}

void VulkanImGuiBackend::renderDrawDataWithSlot(
    VkCommandBuffer commandBuffer,
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDeviceSize bufferMemoryAlignment,
    VulkanImGuiSubmitSlot& slot,
    ImDrawData* drawData)
{
    if (!drawData || drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) {
        return;
    }

    const int fbWidth =
        static_cast<int>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
    const int fbHeight =
        static_cast<int>(drawData->DisplaySize.y * drawData->FramebufferScale.y);
    if (fbWidth <= 0 || fbHeight <= 0) {
        return;
    }

    VkDevice const device = deviceFromContext(m_contextInfo);
    if (!commandBuffer || pipeline == VK_NULL_HANDLE
        || device == VK_NULL_HANDLE
        || pipelineLayout == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] renderDrawDataWithSlot: pipeline/commandBuffer unavailable");
        return;
    }
    const VkDeviceSize alignment = bufferMemoryAlignment > 0 ? bufferMemoryAlignment : 256;

    if (drawData->TotalVtxCount > 0) {
        const VkDeviceSize vertexSize = alignBufferSize(
            static_cast<VkDeviceSize>(drawData->TotalVtxCount) * sizeof(ImDrawVert),
            alignment);
        const VkDeviceSize indexSize = alignBufferSize(
            static_cast<VkDeviceSize>(drawData->TotalIdxCount) * sizeof(ImDrawIdx),
            alignment);

        if (slot.vb == VK_NULL_HANDLE || slot.vbSize < vertexSize) {
            if (!createOrResizeBuffer(
                    slot.vb,
                    slot.vbMemory,
                    slot.vbSize,
                    vertexSize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    alignment)) {
                return;
            }
        }
        if (slot.ib == VK_NULL_HANDLE || slot.ibSize < indexSize) {
            if (!createOrResizeBuffer(
                    slot.ib,
                    slot.ibMemory,
                    slot.ibSize,
                    indexSize,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    alignment)) {
                return;
            }
        }

        ImDrawVert* vtxDst = nullptr;
        ImDrawIdx* idxDst = nullptr;
        if (vkMapMemory(device, slot.vbMemory, 0, vertexSize, 0, reinterpret_cast<void**>(&vtxDst))
            != VK_SUCCESS) {
            return;
        }
        if (vkMapMemory(device, slot.ibMemory, 0, indexSize, 0, reinterpret_cast<void**>(&idxDst))
            != VK_SUCCESS) {
            vkUnmapMemory(device, slot.vbMemory);
            return;
        }

        for (const ImDrawList* drawList : drawData->CmdLists) {
            std::memcpy(
                vtxDst,
                drawList->VtxBuffer.Data,
                static_cast<size_t>(drawList->VtxBuffer.Size) * sizeof(ImDrawVert));
            std::memcpy(
                idxDst,
                drawList->IdxBuffer.Data,
                static_cast<size_t>(drawList->IdxBuffer.Size) * sizeof(ImDrawIdx));
            vtxDst += drawList->VtxBuffer.Size;
            idxDst += drawList->IdxBuffer.Size;
        }

        VkMappedMemoryRange ranges[2]{};
        ranges[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        ranges[0].memory = slot.vbMemory;
        ranges[0].size = VK_WHOLE_SIZE;
        ranges[1].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        ranges[1].memory = slot.ibMemory;
        ranges[1].size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device, 2, ranges);
        vkUnmapMemory(device, slot.vbMemory);
        vkUnmapMemory(device, slot.ibMemory);
    }

    setupRenderStateWithSlot(
        commandBuffer, pipeline, pipelineLayout, slot, drawData, fbWidth, fbHeight);

    const ImVec2 clipOff = drawData->DisplayPos;
    const ImVec2 clipScale = drawData->FramebufferScale;
    VkDescriptorSet lastDescSet = VK_NULL_HANDLE;
    int globalVtxOffset = 0;
    int globalIdxOffset = 0;

    for (const ImDrawList* drawList : drawData->CmdLists) {
        for (int cmdI = 0; cmdI < drawList->CmdBuffer.Size; ++cmdI) {
            const ImDrawCmd* const pcmd = &drawList->CmdBuffer[cmdI];
            if (pcmd->UserCallback != nullptr) {
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState) {
                    setupRenderStateWithSlot(
                        commandBuffer,
                        pipeline,
                        pipelineLayout,
                        slot,
                        drawData,
                        fbWidth,
                        fbHeight);
                } else {
                    pcmd->UserCallback(drawList, pcmd);
                }
                lastDescSet = VK_NULL_HANDLE;
            } else {
                ImVec2 clipMin(
                    (pcmd->ClipRect.x - clipOff.x) * clipScale.x,
                    (pcmd->ClipRect.y - clipOff.y) * clipScale.y);
                ImVec2 clipMax(
                    (pcmd->ClipRect.z - clipOff.x) * clipScale.x,
                    (pcmd->ClipRect.w - clipOff.y) * clipScale.y);

                if (clipMin.x < 0.0f) {
                    clipMin.x = 0.0f;
                }
                if (clipMin.y < 0.0f) {
                    clipMin.y = 0.0f;
                }
                if (clipMax.x > static_cast<float>(fbWidth)) {
                    clipMax.x = static_cast<float>(fbWidth);
                }
                if (clipMax.y > static_cast<float>(fbHeight)) {
                    clipMax.y = static_cast<float>(fbHeight);
                }
                if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
                    continue;
                }

                VkRect2D scissor{};
                scissor.offset.x = static_cast<int32_t>(clipMin.x);
                scissor.offset.y = static_cast<int32_t>(clipMin.y);
                scissor.extent.width = static_cast<uint32_t>(clipMax.x - clipMin.x);
                scissor.extent.height = static_cast<uint32_t>(clipMax.y - clipMin.y);
                vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

                const VkDescriptorSet descSet =
                    reinterpret_cast<VkDescriptorSet>(pcmd->GetTexID());
                if (descSet != lastDescSet) {
                    vkCmdBindDescriptorSets(
                        commandBuffer,
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelineLayout,
                        0,
                        1,
                        &descSet,
                        0,
                        nullptr);
                    lastDescSet = descSet;
                }

                vkCmdDrawIndexed(
                    commandBuffer,
                    pcmd->ElemCount,
                    1,
                    pcmd->IdxOffset + globalIdxOffset,
                    pcmd->VtxOffset + globalVtxOffset,
                    0);
            }
        }
        globalIdxOffset += drawList->IdxBuffer.Size;
        globalVtxOffset += drawList->VtxBuffer.Size;
    }

    VkRect2D fullScissor{ { 0, 0 },
                          { static_cast<uint32_t>(fbWidth), static_cast<uint32_t>(fbHeight) } };
    vkCmdSetScissor(commandBuffer, 0, 1, &fullScissor);
}

bool VulkanImGuiBackend::recordImGuiCommandBuffer(
    VkCommandBuffer commandBuffer,
    std::vector<SecondaryViewportJob>& frameJobs)
{
    if (!commandBuffer || !m_contextInfo.windowContext) {
        return false;
    }

    destroySecondaryViewportJobs(m_pendingSecondaryJobs);
    m_pendingSecondaryJobs.clear();

    VulkanImGuiSubmitSlot& slot = m_submitSlots[m_recordSlotIndex];
    m_recordSlotIndex ^= 1;

    // Official: record main viewport into the app CB first; secondaries go through RenderPlatformWindowsDefault after main Submit.
    // Main uses ping-pong slots; each secondary must use its own ViewportData::RenderBuffers.
    // Pipeline handles come from the job snapshot (event enrich).
    for (SecondaryViewportJob& job : frameJobs) {
        if (!job.drawData) {
            continue;
        }
        if (job.mainViewport) {
            const VkPipeline pipeline = static_cast<VkPipeline>(job.vulkanPipeline);
            const VkPipelineLayout pipelineLayout =
                static_cast<VkPipelineLayout>(job.vulkanPipelineLayout);
            const VkDeviceSize alignment =
                job.vulkanBufferMemoryAlignment > 0
                    ? static_cast<VkDeviceSize>(job.vulkanBufferMemoryAlignment)
                    : 256;
            if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
                LOG_ERROR(
                    "[ImGuiBackend][Vulkan] recordImGuiCommandBuffer: job pipeline snapshot empty");
                continue;
            }
            renderDrawDataWithSlot(
                commandBuffer, pipeline, pipelineLayout, alignment, slot, job.drawData);
            continue;
        }
        m_pendingSecondaryJobs.push_back(job);
        job.drawData = nullptr;
    }

    return true;
}

void VulkanImGuiBackend::submitAndPresentPendingSecondaryViewportJobs()
{
    // Match official: per secondary do RenderWindow + SwapBuffers back-to-back. Avoid submit-all-then-present-all,
    // which can skip Present after NeedRebuild and leave ImageAcquired/RenderComplete hanging.
    if (m_pendingSecondaryJobs.empty()) {
        return;
    }
    if ((m_drawDataCache.configFlags & ImGuiConfigFlags_ViewportsEnable) == 0) {
        return;
    }

    DawnEngine::VulkanWindowContext* const windowContext = m_contextInfo.windowContext;
    if (!windowContext || windowContext->getDevice() == VK_NULL_HANDLE
        || windowContext->getGraphicsQueue() == VK_NULL_HANDLE) {
        return;
    }

    VkDevice const device = windowContext->getDevice();
    VkQueue const queue = windowContext->getGraphicsQueue();

    // If Acquire succeeded (semaphore pending) but recording cannot finish, 0-CB Submit that only waits ImageAcquired to retire it.
    auto retireImageAcquired = [&](VkSemaphore imageAcquired) {
        if (imageAcquired == VK_NULL_HANDLE) {
            return;
        }
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo retireInfo{};
        retireInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        retireInfo.waitSemaphoreCount = 1;
        retireInfo.pWaitSemaphores = &imageAcquired;
        retireInfo.pWaitDstStageMask = &waitStage;
        const VkResult retireErr = vkQueueSubmit(queue, 1, &retireInfo, VK_NULL_HANDLE);
        checkVkResult(retireErr);
    };

    for (SecondaryViewportJob& job : m_pendingSecondaryJobs) {
        if (!job.drawData || job.mainViewport) {
            continue;
        }

        VkPipeline viewportPipeline = static_cast<VkPipeline>(job.vulkanPipelineForViewports);
        if (viewportPipeline == VK_NULL_HANDLE) {
            viewportPipeline = static_cast<VkPipeline>(job.vulkanPipeline);
        }
        const VkPipelineLayout pipelineLayout =
            static_cast<VkPipelineLayout>(job.vulkanPipelineLayout);
        const VkDeviceSize alignment =
            job.vulkanBufferMemoryAlignment > 0
                ? static_cast<VkDeviceSize>(job.vulkanBufferMemoryAlignment)
                : 256;
        if (viewportPipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE) {
            LOG_ERROR(
                "[ImGuiBackend][Vulkan] submitAndPresentPendingSecondaryViewportJobs: "
                "job pipeline snapshot empty");
            continue;
        }

        auto* const vd = static_cast<VkViewportData*>(job.rendererUserData);
        ImGui_ImplVulkanH_Window* const wd = vd ? &vd->Window : nullptr;
        if (!vd || !wd || !job.drawData) {
            continue;
        }

        // --- Pre-draw recreate: custom recreateSecondaryViewportSwapchain (not H_ CreateOrResize) ---
        // While drag-pending: skip recreate (throttle) but still Acquire/record/Present -- old chain may still draw if SUBOPTIMAL;
        // avoid whole-window continue leaving a blank/transparent client; on OUT_OF_DATE skip this frame (GLFW chrome shows size).
        const bool dragResizePending =
            m_secondaryViewportDragResizePending.load(std::memory_order_acquire);
        if ((vd->SwapChainNeedRebuild || vd->SwapChainSuboptimal) && !dragResizePending) {
            const int w = static_cast<int>(job.drawData->DisplaySize.x);
            const int h = static_cast<int>(job.drawData->DisplaySize.y);
            // Minimized / extent<=0: leave flags set; recreate after size recovers next frames.
            if (w <= 0 || h <= 0) {
                continue;
            }
            if (!recreateSecondaryViewportSwapchain(wd, w, h)) {
                LOG_WARNING_FMT(
                    "[ImGuiBackend][Vulkan] secondary recreate failed %dx%d — keep NeedRebuild",
                    w,
                    h);
                continue;
            }
            vd->SwapChainNeedRebuild = false;
            vd->SwapChainSuboptimal = false;
        }

        if (wd->Swapchain == VK_NULL_HANDLE || wd->FrameSemaphores.Size <= 0
            || wd->SemaphoreIndex >= static_cast<uint32_t>(wd->FrameSemaphores.Size)
            || wd->SemaphoreCount == 0) {
            continue;
        }

        ImGui_ImplVulkanH_FrameSemaphores* const fsd = &wd->FrameSemaphores[wd->SemaphoreIndex];
        VkResult err = vkAcquireNextImageKHR(
            device,
            wd->Swapchain,
            UINT64_MAX,
            fsd->ImageAcquiredSemaphore,
            VK_NULL_HANDLE,
            &wd->FrameIndex);
        if (err == VK_ERROR_OUT_OF_DATE_KHR) {
            // Official: no swap this frame; semaphore was not signaled; rebuild at the start of a later frame.
            vd->SwapChainNeedRebuild = true;
            continue;
        }
        if (err == VK_SUBOPTIMAL_KHR) {
            vd->SwapChainSuboptimal = true;
        } else {
            checkVkResult(err);
            if (err != VK_SUCCESS) {
                continue;
            }
        }

        // From here ImageAcquired is pending: any early exit must retire it; bare continue is forbidden.
        if (wd->FrameIndex >= static_cast<uint32_t>(wd->Frames.Size)) {
            retireImageAcquired(fsd->ImageAcquiredSemaphore);
            continue;
        }

        ImGui_ImplVulkanH_Frame* const fd = &wd->Frames[wd->FrameIndex];
        for (;;) {
            err = vkWaitForFences(device, 1, &fd->Fence, VK_TRUE, 100);
            if (err == VK_SUCCESS) {
                break;
            }
            if (err == VK_TIMEOUT) {
                continue;
            }
            checkVkResult(err);
            break;
        }
        if (err != VK_SUCCESS) {
            retireImageAcquired(fsd->ImageAcquiredSemaphore);
            continue;
        }

        err = vkResetCommandPool(device, fd->CommandPool, 0);
        checkVkResult(err);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->CommandBuffer, &beginInfo);
        checkVkResult(err);
        if (err != VK_SUCCESS) {
            retireImageAcquired(fsd->ImageAcquiredSemaphore);
            continue;
        }

        const ImVec4 clearColor = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
        std::memcpy(&wd->ClearValue.color.float32[0], &clearColor, 4 * sizeof(float));

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = wd->RenderPass;
        rpInfo.framebuffer = fd->Framebuffer;
        rpInfo.renderArea.extent.width = static_cast<uint32_t>(wd->Width);
        rpInfo.renderArea.extent.height = static_cast<uint32_t>(wd->Height);
        rpInfo.clearValueCount = (job.flags & ImGuiViewportFlags_NoRendererClear) ? 0 : 1;
        rpInfo.pClearValues =
            (job.flags & ImGuiViewportFlags_NoRendererClear) ? nullptr : &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkWindowRenderBuffers& wrb = vd->RenderBuffers;
        if (wrb.FrameRenderBuffers.Size == 0) {
            wrb.Index = 0;
            wrb.Count = m_minImageCount;
            wrb.FrameRenderBuffers.resize(static_cast<int>(wrb.Count));
            std::memset(
                wrb.FrameRenderBuffers.Data,
                0,
                static_cast<size_t>(wrb.FrameRenderBuffers.size_in_bytes()));
        }
        wrb.Index = (wrb.Index + 1) % wrb.Count;
        VkFrameRenderBuffers& rb = wrb.FrameRenderBuffers[static_cast<int>(wrb.Index)];
        renderDrawDataWithOfficialFrameBuffers(
            fd->CommandBuffer,
            viewportPipeline,
            pipelineLayout,
            alignment,
            rb.VertexBufferMemory,
            rb.IndexBufferMemory,
            rb.VertexBufferSize,
            rb.IndexBufferSize,
            rb.VertexBuffer,
            rb.IndexBuffer,
            job.drawData);
        vkCmdEndRenderPass(fd->CommandBuffer);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &fsd->ImageAcquiredSemaphore;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &fd->CommandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &fsd->RenderCompleteSemaphore;
        err = vkEndCommandBuffer(fd->CommandBuffer);
        checkVkResult(err);
        if (err != VK_SUCCESS) {
            retireImageAcquired(fsd->ImageAcquiredSemaphore);
            continue;
        }
        err = vkResetFences(device, 1, &fd->Fence);
        checkVkResult(err);
        err = vkQueueSubmit(queue, 1, &submitInfo, fd->Fence);
        checkVkResult(err);
        if (err != VK_SUCCESS) {
            // ImageAcquired may still be pending (Submit did not consume it); try retire again.
            retireImageAcquired(fsd->ImageAcquiredSemaphore);
            continue;
        }

        // --- SwapBuffers: Present this window immediately, then the next ---
        uint32_t presentIndex = wd->FrameIndex;
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &fsd->RenderCompleteSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &wd->Swapchain;
        presentInfo.pImageIndices = &presentIndex;
        err = vkQueuePresentKHR(queue, &presentInfo);
        if (err == VK_ERROR_OUT_OF_DATE_KHR) {
            vd->SwapChainNeedRebuild = true;
        } else if (err == VK_SUBOPTIMAL_KHR) {
            vd->SwapChainSuboptimal = true;
        } else {
            checkVkResult(err);
        }
        // After Submit+Present are queued, always advance the index. Official OUT_OF_DATE skips advance because the next
        // CreateOrResize+WaitIdle destroys the old semaphore. Without recreate, failing to advance reuses the same
        // ImageAcquired that still has an unfinished Submit wait -> VUID-01779.
        wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount;
    }
}

void VulkanImGuiBackend::submitAndPresentSecondaryViewportWindows()
{
    // Match official RenderPlatformWindowsDefault: after main Submit, per-window RenderWindow+SwapBuffers.
    submitAndPresentPendingSecondaryViewportJobs();
    destroySecondaryViewportJobs(m_pendingSecondaryJobs);
    m_pendingSecondaryJobs.clear();
}

bool VulkanImGuiBackend::NewFrame()
{
    // Phase gating lives only in PrepareDrawData: do not gate or early-return here; always open the frame.
    // Returning false on Render phase here skips ImGui::NewFrame/atlas updates and, combined with
    // event-side upload then SetStatus(OK), easily yields missing glyphs (Vulkan hit this before).
    // TEMP re-mark WantUpdates was unstable (suspected driver/system hang); disabled -- this backend must wait for render.
    // tryRemakeUntakenJobTexturesWantUpdates();
    ImGui::SetCurrentContext(m_imguiContext);
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    return true;
}

void VulkanImGuiBackend::PrepareDrawData()
{
    // -- Policy (current) --
    // DX11/DX12/OpenGL: event need not wait for render (validated). Vulkan: re-marking WantUpdates was unstable (suspected driver hang),
    // so the event thread must wait until render finishes the frame before composing the next -- upload CBs must be Submitted before jobs are dropped.
    // applyOfficial* must still clear produced jobs to avoid secondary old-swapchain UAF (see ImGuiBackend).
    //
    // -- Hard rule: UpdatePlatformWindows stays on the event side --
    // Official example Updates after main FrameRender(Submit). This engine's render thread does not call ImGui,
    // so full-speed Update runs in event Prepare (base / early-out below). Secondary Resize -> onSecondary ->
    // applyOfficial only marks pending; Prepare-end flush on mouse-up then marks NeedRebuild.
    // Do not move Update after main Submit (fights the main graphics queue). Texture record upload still must be after ImGui::Render.
    //
    // -- Snapshot decoupling: why Render can ignore the event thread --
    // copyViewportDrawData enqueues deep-copied snapshots; except for secondary Create/Destroy/Resize,
    // render only needs to take jobs and can finish drawing without caring what the event thread is composing.
    // Independent secondary lifetime changes still handshake at render start (ImGuiBackend::onSecondaryViewportResourceChanged
    // -> applyPendingImGuiSecondaryViewportChanges) so Destroy/Create/Resize land at a safe point,
    // avoiding parallel UAF when event mutates HWND/Swapchain while render holds old resources; with handshake + snapshots, Render just take-and-draw.
    //
    // -- Dual-thread handoff (weight ~6:4) --
    // [~60%] Wait while phase is Render first: if render is drawing an independent secondary, this event frame must not
    // EndFrame/UpdatePlatformWindows and Destroy it, or render crashes on stale HWND/Swapchain;
    // also blocks flipping to Event before render returns the baton then push-dropping unconsumed jobs.
    waitWhileTextureUploadPhaseIsRender();

    // [~40%] Extra release: empty queue, or render already took/drew/returned -> force Event
    // so the event thread can enter next Prepare (compose/upload/enqueue); together with the wait above, no lost baton/frames.
    if (secondaryViewportJobsEmptyForEventThread()) {
        m_phase.store(ImGuiTextureUploadPhase::Event, std::memory_order_release);
    }

    // Non-Event: this round does not hand off (jobs still pending, etc.), but NewFrame+UI already opened a frame --
    // must finish the event-thread frame tail: EndFrame; with Viewports also UpdatePlatformWindows,
    // else next NewFrame asserts FrameCount != FrameCountPlatformEnded; then unbind GImGui.
    // (UpdatePlatformWindows stays event-side: see hard rule above.)
    if (m_phase.load(std::memory_order_acquire) != ImGuiTextureUploadPhase::Event) {
        ImGui::EndFrame();
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            flushDeferredSecondaryViewportSwapchainRebuild();
        }
        ImGui::SetCurrentContext(nullptr);
        return;
    }

    // Event path: compose (incl. full-speed UpdatePlatformWindows) -> record upload first half -> copy enqueue -> hand off.
    // Do not call official UpdateTexture on the event side (Submit+WaitIdle fights the render queue).
    ImGuiBackendBase::PrepareDrawData();
    if (!m_drawData) {
        LOG_ERROR("[ImGuiBackend][Vulkan] PrepareDrawData: ImGui::GetDrawData returned null");
    }

    // Pipeline handles are written into each job by enrichSecondaryViewportJob.
    // Upload pool/CB are long-lived members; only this frame's staging rides on the main-viewport job.
    std::vector<VulkanTexUploadStaging> uploadStagings;
    const bool recordedUploads =
        recordPendingTextureUploadsOnEventPhase(uploadStagings);

    // Deep-copy into the queue; attach staging when recording happened (other backends pass empty).
    if (recordedUploads) {
        copyViewportDrawData(std::move(uploadStagings));
    } else {
        copyViewportDrawData();
    }

    ImGui::SetCurrentContext(nullptr);
    markSecondaryViewportJobsRenderSideHasData();
    // markEventRenderHandoffEventPrepareComplete();
    m_phase.store(ImGuiTextureUploadPhase::Render, std::memory_order_release);
}

void VulkanImGuiBackend::Render(void* context)
{
    // Snapshots are decoupled: once taken, draw to completion; no need to sync event-thread compose content.
    // UpdatePlatformWindows already finished in event Prepare (hard rule: do not move after main Submit and fight the queue).
    // Independent secondary Create/Destroy/Resize land via apply handshake (see ImGuiBackend
    // onSecondaryViewportResourceChanged); this path only consumes jobs.
    // Match official FrameRender RenderDrawData: record main into the main CB only; secondaries wait for post-Submit
    // submitAndPresent (~ RenderPlatformWindowsDefault). After record/return, phase->Event.
    auto* const commandBuffer = static_cast<VkCommandBuffer>(context);
    if (!commandBuffer) {
        LOG_ERROR("[ImGuiBackend][Vulkan] Render: command buffer is null");
        m_phase.store(ImGuiTextureUploadPhase::Event, std::memory_order_release);
        return;
    }

    std::vector<SecondaryViewportJob> frameJobs;
    if (!takeFrontSecondaryViewportJobs(frameJobs)) {
        tryMarkSecondaryViewportJobsRenderSideTakeMissed();
        m_phase.store(ImGuiTextureUploadPhase::Event, std::memory_order_release);
        return;
    }

    // Texture upload second half: Submit the member CB first (must finish before draw sampling); skip if already submitted.
    for (SecondaryViewportJob& job : frameJobs) {
        if (job.mainViewport) {
            submitPendingTextureUploadsOnRenderPhase(job);
            break;
        }
    }

    for (SecondaryViewportJob& job : frameJobs) {
        if (job.drawData) {
            job.drawData->Textures = nullptr;
        }
    }

    if (!recordImGuiCommandBuffer(commandBuffer, frameJobs)) {
        LOG_ERROR("[ImGuiBackend][Vulkan] Render: recordImGuiCommandBuffer failed");
    }
    // Secondaries moved into m_pendingSecondaryJobs; return remaining main jobs (align DX11).
    returnSecondaryViewportJobs(std::move(frameJobs));
    // tryMarkEventRenderHandoffRenderComplete();
    // Return the baton after main record/return; secondary submitAndPresent still runs from endFrame after main Submit.
    m_phase.store(ImGuiTextureUploadPhase::Event, std::memory_order_release);
}

void VulkanImGuiBackend::SyncFromEngineContext(void* contextInfo)
{
    // Align DX11/DX12: only rebuild Renderer-side objects that depend on the new RenderPass / swapchain.
    // Do not call ImGui_ImplGlfw_Shutdown / DestroyPlatformWindows -- that clears secondary PlatformHandle
    // while ImGui's viewport list remains -> next-frame UpdateMouseData null-handle crash.
    if (contextInfo) {
        m_contextInfo = *static_cast<VulkanImGuiBackendContextInfo*>(contextInfo);
    }

    if (m_imguiContext == nullptr || m_contextInfo.windowContext == nullptr) {
        return;
    }

    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    if (windowCtx->getDevice() == VK_NULL_HANDLE || windowCtx->getRenderPass() == VK_NULL_HANDLE) {
        LOG_ERROR("[ImGuiBackend][Vulkan] SyncFromEngineContext: device/renderPass is null");
        return;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    if (windowCtx->getWindow() != nullptr) {
        glfwGetFramebufferSize(windowCtx->getWindow(), &framebufferWidth, &framebufferHeight);
    }
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        return;
    }

    clearCopiedViewportDrawData();
    destroySecondaryViewportJobs(m_pendingSecondaryJobs);
    m_pendingSecondaryJobs.clear();

    ImGui::SetCurrentContext(m_imguiContext);
    waitDeviceIdle();
    destroySubmitSlots();

    auto* const bd = static_cast<VkBackendData*>(ImGui::GetIO().BackendRendererUserData);
    if (bd == nullptr) {
        LOG_ERROR("[ImGuiBackend][Vulkan] SyncFromEngineContext: BackendRendererUserData null");
        ImGui::SetCurrentContext(nullptr);
        return;
    }

    m_minImageCount = (std::max)(2u, windowCtx->getSwapChainImageCount());
    bd->VulkanInitInfo.MinImageCount = m_minImageCount;
    bd->VulkanInitInfo.ImageCount = m_minImageCount;
    bd->VulkanInitInfo.Queue = windowCtx->getGraphicsQueue();
    bd->VulkanInitInfo.DescriptorPool = windowCtx->getDescriptorPool();
    bd->VulkanInitInfo.PipelineCache = windowCtx->getPipelineCache();

    ImGui_ImplVulkan_PipelineInfo pipelineInfo = bd->VulkanInitInfo.PipelineInfoMain;
    pipelineInfo.RenderPass = windowCtx->getRenderPass();
    pipelineInfo.Subpass = 0;
    pipelineInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_CreateMainPipeline(&pipelineInfo);

    if (!createSubmitSlots()) {
        LOG_ERROR("[ImGuiBackend][Vulkan] SyncFromEngineContext: rebuild submit slots failed");
        ImGui::SetCurrentContext(nullptr);
        return;
    }

    ImGui::SetCurrentContext(nullptr);
}

void VulkanImGuiBackend::abandonPendingTextureUploads()
{
    // Upload resources ride on jobs; caller should clearCopiedViewportDrawData / destroy pending first.
    m_phase.store(ImGuiTextureUploadPhase::Event, std::memory_order_release);
}

void VulkanImGuiBackend::discardDeferredPresentSubmissions()
{
    destroySecondaryViewportJobs(m_pendingSecondaryJobs);
    m_pendingSecondaryJobs.clear();
    clearCopiedViewportDrawData();
    abandonPendingTextureUploads();
}

void* VulkanImGuiBackend::registerOffscreenTexture(VkImageView imageView, VkImageLayout layout)
{
    if (m_imguiContext == nullptr || imageView == VK_NULL_HANDLE) {
        return nullptr;
    }

    ImGui::SetCurrentContext(m_imguiContext);
    if (ImGui::GetIO().BackendRendererUserData == nullptr
        || m_offscreenTextureSampler == VK_NULL_HANDLE) {
        return nullptr;
    }

    const VkDescriptorSet descriptorSet =
        ImGui_ImplVulkan_AddTexture(m_offscreenTextureSampler, imageView, layout);
    if (descriptorSet == VK_NULL_HANDLE) {
        LOG_ERROR(
            "[ImGuiBackend][Vulkan] registerOffscreenTexture: AddTexture failed "
            "(descriptor pool exhausted or invalid view)");
        return nullptr;
    }
    return reinterpret_cast<void*>(descriptorSet);
}

void VulkanImGuiBackend::unregisterOffscreenTexture(void* descriptorSet)
{
    if (m_imguiContext == nullptr || !descriptorSet) {
        return;
    }

    ImGui::SetCurrentContext(m_imguiContext);
    if (ImGui::GetIO().BackendRendererUserData == nullptr) {
        return;
    }

    ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(descriptorSet));
}

bool VulkanImGuiBackend::createOffscreenTextureSampler()
{
    if (m_offscreenTextureSampler != VK_NULL_HANDLE) {
        return true;
    }

    VkDevice const device =
        m_contextInfo.windowContext ? m_contextInfo.windowContext->getDevice() : VK_NULL_HANDLE;
    if (device == VK_NULL_HANDLE) {
        return false;
    }

    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.minLod = -1000;
    info.maxLod = 1000;
    info.maxAnisotropy = 1.0f;
    if (vkCreateSampler(device, &info, nullptr, &m_offscreenTextureSampler) != VK_SUCCESS) {
        LOG_ERROR("[ImGuiBackend][Vulkan] createOffscreenTextureSampler: vkCreateSampler failed");
        return false;
    }
    return true;
}

void VulkanImGuiBackend::Shutdown()
{
    if (m_imguiContext == nullptr) {
        return;
    }

    //uninstallPlatformViewportHooks();
    uninstallRendererViewportHooks();
    destroySecondaryViewportJobs(m_pendingSecondaryJobs);
    m_pendingSecondaryJobs.clear();
    clearCopiedViewportDrawData();
    waitDeviceIdle();
    destroySubmitSlots();
    destroyTextureUploadCommandResources();

    ImGui::SetCurrentContext(m_imguiContext);
    shutdownPlatformAndRenderer();

    DawnEngine::VulkanWindowContext* const windowCtx = m_contextInfo.windowContext;
    VkDevice const device = windowCtx ? windowCtx->getDevice() : VK_NULL_HANDLE;
    if (m_offscreenTextureSampler != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        vkDestroySampler(device, m_offscreenTextureSampler, nullptr);
        m_offscreenTextureSampler = VK_NULL_HANDLE;
    }

    clearCopiedViewportDrawData();
    DestroyImGuiBackendContext(m_imguiContext);
    m_imguiContext = nullptr;
    m_drawData = nullptr;

    LOG_INFO("[ImGuiBackend][Vulkan] shutdown");
}

} // namespace ImGuiWidgets
