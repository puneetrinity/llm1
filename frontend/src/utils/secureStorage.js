// frontend/src/utils/secureStorage.js - Secure storage utility with expiration

/**
 * Secure storage utility that wraps localStorage with error handling and expiration
 */
export const secureStorage = {
  /**
   * Store an item with optional expiration
   * @param {string} key - Storage key
   * @param {any} value - Value to store
   * @param {number} ttlHours - Time to live in hours (default: 24)
   */
  setItem: (key, value, ttlHours = 24) => {
    try {
      // In production, consider encrypting the value here
      const item = {
        value,
        timestamp: Date.now(),
        expires: Date.now() + (ttlHours * 60 * 60 * 1000)
      };
      localStorage.setItem(key, JSON.stringify(item));
    } catch (error) {
      console.warn('Failed to store item:', error);
    }
  },

  /**
   * Retrieve an item, checking for expiration
   * @param {string} key - Storage key
   * @returns {any|null} Stored value or null if not found/expired
   */
  getItem: (key) => {
    try {
      const itemStr = localStorage.getItem(key);
      if (!itemStr) return null;

      const item = JSON.parse(itemStr);
      
      // Check if expired
      if (item.expires && Date.now() > item.expires) {
        localStorage.removeItem(key);
        return null;
      }

      return item.value;
    } catch (error) {
      console.warn('Failed to retrieve item:', error);
      localStorage.removeItem(key); // Clean up corrupted data
      return null;
    }
  },

  /**
   * Remove an item from storage
   * @param {string} key - Storage key
   */
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.warn('Failed to remove item:', error);
    }
  },

  /**
   * Clear all expired items from storage
   */
  clearExpired: () => {
    try {
      const keysToRemove = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key) {
          const itemStr = localStorage.getItem(key);
          try {
            const item = JSON.parse(itemStr);
            if (item.expires && Date.now() > item.expires) {
              keysToRemove.push(key);
            }
          } catch {
            // Not a secure storage item, skip
          }
        }
      }
      
      keysToRemove.forEach(key => localStorage.removeItem(key));
      
      if (keysToRemove.length > 0) {
        console.log(`Cleared ${keysToRemove.length} expired items from storage`);
      }
    } catch (error) {
      console.warn('Failed to clear expired items:', error);
    }
  },

  /**
   * Get storage statistics
   * @returns {Object} Storage usage statistics
   */
  getStats: () => {
    try {
      let totalItems = 0;
      let expiredItems = 0;
      let validItems = 0;

      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key) {
          const itemStr = localStorage.getItem(key);
          try {
            const item = JSON.parse(itemStr);
            if (item.expires) {
              totalItems++;
              if (Date.now() > item.expires) {
                expiredItems++;
              } else {
                validItems++;
              }
            }
          } catch {
            // Not a secure storage item, skip
          }
        }
      }

      return {
        totalItems,
        expiredItems,
        validItems,
        storageUsed: new Blob(Object.values(localStorage)).size
      };
    } catch (error) {
      console.warn('Failed to get storage stats:', error);
      return { totalItems: 0, expiredItems: 0, validItems: 0, storageUsed: 0 };
    }
  }
};

// Auto-cleanup expired items on module load
secureStorage.clearExpired();
