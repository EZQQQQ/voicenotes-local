local M = {}

local menubar = hs.menubar.new()
local watcher = nil
local hotkey = nil

local function run(command)
  local output, status = hs.execute(command, true)
  if status then
    return output
  end
  return nil
end

local function json(command)
  local output = run(command)
  if not output then
    return nil
  end
  return hs.json.decode(output)
end

local function setTitle(status)
  if not menubar then
    return
  end
  local label = "VN"
  if status and status.state_label == "recording" then
    label = "VN REC"
  elseif status and status.state_label == "processing" then
    label = "VN RUN"
  elseif status and status.state_label == "queued" then
    label = "VN Q"
  elseif status and status.state_label == "error" then
    label = "VN !"
  end
  menubar:setTitle(label)
end

local function refresh()
  setTitle(json("voicenotes status --json"))
end

local function outputRoot()
  local config = json("voicenotes config --json")
  if config and config.output_root then
    return config.output_root
  end
  return os.getenv("HOME") .. "/VoiceNotes"
end

local function shellQuote(value)
  return "'" .. tostring(value):gsub("'", "'\\''") .. "'"
end

local function menuItems()
  local status = json("voicenotes status --json")
  local toggleTitle = "Start Recording"
  if status and status.state_label == "recording" then
    toggleTitle = "Stop Recording"
  end

  return {
    {title = toggleTitle, fn = M.toggle},
    {title = "Open VoiceNotes Folder", fn = function()
      hs.execute("open -g " .. shellQuote(outputRoot()), true)
    end},
    {title = "-"},
    {title = "Quit", fn = function()
      if watcher then
        watcher:stop()
        watcher = nil
      end
      if hotkey then
        hotkey:disable()
        hotkey = nil
      end
      if menubar then
        menubar:delete()
        menubar = nil
      end
      local app = hs.application.get("Hammerspoon")
      if app then
        app:kill()
      end
    end},
  }
end

function M.toggle()
  hs.task.new("/bin/zsh", function()
    refresh()
  end, {"-lc", "voicenotes toggle"}):start()
end

function M.start()
  hs.execute("mkdir -p ~/.voicenotes/run", true)
  local config = json("voicenotes config --json")
  local mods = {"cmd"}
  local key = "`"
  if config and config.hotkey then
    mods = config.hotkey.mods or mods
    key = config.hotkey.key or key
  end
  hotkey = hs.hotkey.bind(mods, key, M.toggle)
  menubar:setMenu(menuItems)
  refresh()
  watcher = hs.pathwatcher.new(os.getenv("HOME") .. "/.voicenotes/run", refresh)
  watcher:start()
end

M.start()

return M
