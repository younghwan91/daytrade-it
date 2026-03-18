import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/signals', label: 'Signals', icon: '📡' },
  { to: '/trades', label: 'Trades', icon: '💹' },
  { to: '/portfolio', label: 'Portfolio', icon: '💼' },
  { to: '/backtest', label: 'Backtest', icon: '🔬' },
]

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={`${
        collapsed ? 'w-16' : 'w-64'
      } bg-gray-900 text-white flex flex-col transition-all duration-300`}
    >
      {/* Logo/App Name */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        {!collapsed && <h1 className="text-xl font-bold">GPT Quant</h1>}
        {collapsed && <span className="text-xl font-bold mx-auto">GQ</span>}
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-2">
        <ul className="space-y-1">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-gray-800'
                  }`
                }
              >
                <span className="text-lg">{item.icon}</span>
                {!collapsed && <span>{item.label}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Collapse Button */}
      <div className="p-2 border-t border-gray-700">
        <button
          onClick={onToggle}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-gray-300 hover:bg-gray-800 transition-colors"
        >
          <span className="text-lg">{collapsed ? '→' : '←'}</span>
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  )
}
